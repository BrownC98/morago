//! macOS system audio through a Core Audio process tap (macOS 14.6+).
//!
//! Same method as cpal's loopback (cpal 0.18, Apache-2.0 / MIT, `coreaudio/macos/loopback.rs`):
//! a tap on the default output device, read through a private aggregate device. cpal builds it
//! only when the output device has no input channels. A full-duplex device (AirPods, a USB
//! headset, an audio interface) would give the microphone instead, so this module builds the
//! tap itself and reads it with an IOProc. Needs NSAudioCaptureUsageDescription (Info.plist).
//!
//! Not testable without a Mac. CI compiles it; see the manual test steps in the task result.

use super::pipeline::{TapBuffer, host_lag, tap_frames, tap_frames_to_mono};
use super::worker::Feeder;
use super::{Event, MAX_CAPTURE_LAG, Running};
use crossbeam_channel::Sender;
use objc2::{AnyThread, rc::Retained};
use objc2_core_audio::{
    AudioConvertHostTimeToNanos, AudioDeviceCreateIOProcID, AudioDeviceDestroyIOProcID,
    AudioDeviceIOProcID, AudioDeviceStart, AudioDeviceStop, AudioGetCurrentHostTime,
    AudioHardwareCreateAggregateDevice, AudioHardwareCreateProcessTap,
    AudioHardwareDestroyAggregateDevice, AudioHardwareDestroyProcessTap,
    AudioObjectAddPropertyListener, AudioObjectGetPropertyData, AudioObjectID,
    AudioObjectPropertyAddress, AudioObjectRemovePropertyListener, CATapDescription,
    CATapMuteBehavior, kAudioAggregateDeviceNameKey, kAudioAggregateDeviceTapAutoStartKey,
    kAudioAggregateDeviceTapListKey, kAudioAggregateDeviceUIDKey,
    kAudioDevicePropertyDeviceIsAlive, kAudioDevicePropertyDeviceUID,
    kAudioDevicePropertyNominalSampleRate, kAudioEndPointDeviceIsPrivateKey,
    kAudioHardwarePropertyDefaultOutputDevice, kAudioObjectPropertyElementMain,
    kAudioObjectPropertyScopeGlobal, kAudioObjectSystemObject, kAudioSubTapDriftCompensationKey,
    kAudioSubTapUIDKey, kAudioTapPropertyFormat,
};
use objc2_core_audio_types::{
    AudioBufferList, AudioStreamBasicDescription, AudioTimeStamp, AudioTimeStampFlags,
    kAudioFormatLinearPCM,
};
use objc2_core_foundation::{
    CFArray, CFDictionary, CFMutableDictionary, CFRetained, CFString, kCFAllocatorDefault,
    kCFTypeArrayCallBacks, kCFTypeDictionaryKeyCallBacks, kCFTypeDictionaryValueCallBacks,
};
use objc2_foundation::{NSArray, NSNumber, NSString};
use std::ffi::{CStr, c_void};
use std::mem::MaybeUninit;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

type OSStatus = i32;

static INSTANCE: AtomicU32 = AtomicU32::new(0);

/// `kAudioFormatFlagIsFloat` (CoreAudioBaseTypes.h).
const FLAG_FLOAT: u32 = 1;
/// Buffers of one IOProc call that are mixed. A tap has one or two.
const MAX_BUFFERS: usize = 8;

/// What the IOProc needs. Lives in a `Box` that `TapStream` frees after the IOProc is gone.
struct IoContext {
    feeder: Feeder,
    rate: u32,
}

/// What the property listener needs.
struct ListenerContext {
    events: Sender<Event>,
    generation: u64,
}

/// A running tap. Dropping it stops the IOProc and destroys the aggregate device and the tap.
pub struct TapStream {
    tap_id: AudioObjectID,
    aggregate_id: Option<AudioObjectID>,
    proc_id: AudioDeviceIOProcID,
    started: bool,
    context: *mut IoContext,
    listener: *mut ListenerContext,
    listening: bool,
    /// The default output device that the tap listens to.
    output_device: AudioObjectID,
    /// Sample rate of the tap and of the output device when the tap was opened.
    rate: u32,
    nominal_rate: f64,
}

// The raw pointers are owned by this value and used by Core Audio only until `drop` stops them.
unsafe impl Send for TapStream {}

impl Running for TapStream {
    /// A change that raises no stream error (the listeners also report most of them): another
    /// default output (AirPods connected), another sample rate (AirPods change to the headset
    /// profile), or a dead device. The tap would give silence or data at the wrong rate.
    fn stale(&self) -> bool {
        let moved = default_output_device().is_ok_and(|id| id != self.output_device);
        let tap_rate = tap_format(self.tap_id).map(|f| f.mSampleRate as u32).ok();
        let rate_changed = tap_rate.is_some_and(|r| r != self.rate)
            || device_nominal_rate(self.output_device).is_some_and(|r| r != self.nominal_rate);
        moved || rate_changed || !device_alive(self.output_device)
    }
}

impl Drop for TapStream {
    fn drop(&mut self) {
        unsafe {
            if self.listening {
                for selector in LISTENED {
                    let _ = AudioObjectRemovePropertyListener(
                        self.output_device,
                        NonNull::from(&address(selector)),
                        Some(on_device_change),
                        self.listener.cast::<c_void>(),
                    );
                }
            }
            if let Some(aggregate) = self.aggregate_id {
                if self.started {
                    let _ = AudioDeviceStop(aggregate, self.proc_id);
                }
                if self.proc_id.is_some() {
                    let _ = AudioDeviceDestroyIOProcID(aggregate, self.proc_id);
                }
                let _ = AudioHardwareDestroyAggregateDevice(aggregate);
            }
            let _ = AudioHardwareDestroyProcessTap(self.tap_id);
            if !self.context.is_null() {
                drop(Box::from_raw(self.context));
            }
            if !self.listener.is_null() {
                drop(Box::from_raw(self.listener));
            }
        }
    }
}

/// Property changes of the output device that end a tap.
const LISTENED: [u32; 2] = [
    kAudioDevicePropertyNominalSampleRate,
    kAudioDevicePropertyDeviceIsAlive,
];

pub fn open(feeder: Feeder, events: Sender<Event>, generation: u64) -> anyhow::Result<TapStream> {
    let output_device = default_output_device()?;
    let uid = device_uid(output_device)?;
    let pid = std::process::id();
    let instance = INSTANCE.fetch_add(1, Ordering::Relaxed);

    // 1. The tap: all processes, on the default output device. Audio still reaches the speakers.
    let tap_desc = unsafe {
        CATapDescription::initWithProcesses_andDeviceUID_withStream(
            CATapDescription::alloc(),
            &NSArray::new(),
            &uid,
            0,
        )
    };
    unsafe {
        tap_desc.setMuteBehavior(CATapMuteBehavior::Unmuted);
        tap_desc.setName(&NSString::from_str(&format!(
            "morago system audio {pid}.{instance}"
        )));
        tap_desc.setPrivate(true);
        tap_desc.setExclusive(true); // the (empty) process list is the exclusion list
    }
    let mut tap_id: MaybeUninit<AudioObjectID> = MaybeUninit::uninit();
    let status =
        unsafe { AudioHardwareCreateProcessTap(Some(tap_desc.as_ref()), tap_id.as_mut_ptr()) };
    check(status, "process tap")?;
    let tap_id = unsafe { tap_id.assume_init() };

    // From here on, `stream` destroys what exists if a later step fails.
    let mut stream = TapStream {
        tap_id,
        aggregate_id: None,
        proc_id: None,
        started: false,
        context: std::ptr::null_mut(),
        listener: std::ptr::null_mut(),
        listening: false,
        output_device,
        rate: 0,
        nominal_rate: device_nominal_rate(output_device).unwrap_or(0.0),
    };

    // 2. The format of the tap (rate and layout).
    let format = tap_format(tap_id)?;
    if format.mFormatID != kAudioFormatLinearPCM
        || format.mFormatFlags & FLAG_FLOAT == 0
        || format.mBitsPerChannel != 32
        || format.mSampleRate.is_nan()
        || format.mSampleRate < 1.0
    {
        log::error!("audio System: unexpected tap format {format:?}");
        anyhow::bail!("시스템 소리의 형식을 읽을 수 없습니다.");
    }
    stream.rate = format.mSampleRate as u32;
    log::info!(
        "audio System: tap {} Hz, {} ch",
        format.mSampleRate,
        format.mChannelsPerFrame
    );

    // 3. A private aggregate device that holds the tap.
    let tap_uid = unsafe { tap_desc.UUID().UUIDString() };
    let properties = aggregate_properties(
        tap_uid,
        &format!("com.morago.SystemAudioAggregate.{pid}.{instance}"),
        &format!("Morago system audio {pid}.{instance}"),
    );
    let mut aggregate_id: AudioObjectID = 0;
    let status = unsafe {
        AudioHardwareCreateAggregateDevice(properties.as_ref(), NonNull::from(&mut aggregate_id))
    };
    check(status, "aggregate device")?;
    stream.aggregate_id = Some(aggregate_id);

    // 4. Tell the supervisor when the output device changes its rate or dies.
    stream.listener = Box::into_raw(Box::new(ListenerContext { events, generation }));
    for selector in LISTENED {
        let status = unsafe {
            AudioObjectAddPropertyListener(
                output_device,
                NonNull::from(&address(selector)),
                Some(on_device_change),
                stream.listener.cast::<c_void>(),
            )
        };
        check(status, "property listener")?;
        stream.listening = true; // Drop removes both; removing an unset one only returns an error
    }

    // 5. Read the aggregate device with an IOProc.
    stream.context = Box::into_raw(Box::new(IoContext {
        feeder,
        rate: stream.rate,
    }));
    let mut proc_id: AudioDeviceIOProcID = None;
    let status = unsafe {
        AudioDeviceCreateIOProcID(
            aggregate_id,
            Some(io_proc),
            stream.context.cast::<c_void>(),
            NonNull::from(&mut proc_id),
        )
    };
    check(status, "IOProc")?;
    stream.proc_id = proc_id;
    check(unsafe { AudioDeviceStart(aggregate_id, proc_id) }, "start")?;
    stream.started = true;
    Ok(stream)
}

/// Called by Core Audio when the output device changes its rate or dies. Sends one event.
unsafe extern "C-unwind" fn on_device_change(
    _object: AudioObjectID,
    _count: u32,
    _addresses: NonNull<AudioObjectPropertyAddress>,
    client: *mut c_void,
) -> OSStatus {
    // SAFETY: `client` is the `ListenerContext` box of the `TapStream`, freed after the
    // listeners are removed.
    let ctx = unsafe { &*(client as *const ListenerContext) };
    let _ = ctx.events.try_send(Event {
        generation: ctx.generation,
        fatal: true,
        text: "the output device changed its sample rate or stopped".into(),
    });
    0
}

/// Called by Core Audio on its IO thread: queue the mono block, nothing else. No log, no
/// allocation, no panic path.
unsafe extern "C-unwind" fn io_proc(
    _device: AudioObjectID,
    _now: NonNull<AudioTimeStamp>,
    input: NonNull<AudioBufferList>,
    input_time: NonNull<AudioTimeStamp>,
    _output: NonNull<AudioBufferList>,
    _output_time: NonNull<AudioTimeStamp>,
    client: *mut c_void,
) -> OSStatus {
    // SAFETY: `client` is the `IoContext` box of the `TapStream`, freed only after the IOProc is
    // stopped and destroyed. Core Audio gives a valid buffer list for the call.
    let ctx = unsafe { &*(client as *const IoContext) };
    let list = unsafe { input.as_ref() };
    let count = (list.mNumberBuffers as usize).min(MAX_BUFFERS);
    let raw = unsafe { std::slice::from_raw_parts(list.mBuffers.as_ptr(), count) };
    let mut views = [TapBuffer {
        channels: 0,
        data: &[],
    }; MAX_BUFFERS];
    for (view, b) in views.iter_mut().zip(raw) {
        if !b.mData.is_null() && b.mNumberChannels > 0 {
            view.channels = b.mNumberChannels as usize;
            // SAFETY: Core Audio gives `mDataByteSize` valid bytes of f32 samples at `mData`.
            view.data = unsafe {
                std::slice::from_raw_parts(b.mData as *const f32, b.mDataByteSize as usize / 4)
            };
        }
    }
    let views = &views[..count];
    let frames = tap_frames(views);
    if frames == 0 {
        return 0;
    }
    let now = Instant::now();
    let time = unsafe { input_time.as_ref() };
    let lag = if time.mFlags.contains(AudioTimeStampFlags::HostTimeValid) {
        // SAFETY: plain clock reads and a unit conversion.
        let (now_ns, then_ns) = unsafe {
            (
                AudioConvertHostTimeToNanos(AudioGetCurrentHostTime()),
                AudioConvertHostTimeToNanos(time.mHostTime),
            )
        };
        host_lag(now_ns, then_ns, MAX_CAPTURE_LAG)
    } else {
        None
    };
    let lag = lag.unwrap_or_else(|| {
        Duration::try_from_secs_f64(frames as f64 / f64::from(ctx.rate)).unwrap_or_default()
    });
    let start = now.checked_sub(lag).unwrap_or(now);
    ctx.feeder
        .feed(start, ctx.rate, |mono| tap_frames_to_mono(views, mono));
    0
}

/// Korean message for a Core Audio call that failed. The most likely cause is the permission.
fn check(status: OSStatus, what: &str) -> anyhow::Result<()> {
    if status == 0 {
        return Ok(());
    }
    log::error!("audio System: {what} failed, OSStatus {status}");
    anyhow::bail!(
        "시스템 소리를 시작하지 못했습니다. 시스템 설정 > 개인정보 보호 및 보안 > 화면 및 시스템 오디오 녹음에서 morago를 켜고 앱을 다시 시작하세요."
    )
}

fn address(selector: u32) -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress {
        mSelector: selector,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain,
    }
}

fn default_output_device() -> anyhow::Result<AudioObjectID> {
    let mut id: AudioObjectID = 0;
    let mut size = size_of::<AudioObjectID>() as u32;
    let property = address(kAudioHardwarePropertyDefaultOutputDevice);
    let status = unsafe {
        AudioObjectGetPropertyData(
            kAudioObjectSystemObject as AudioObjectID,
            NonNull::from(&property),
            0,
            std::ptr::null(),
            NonNull::from(&mut size),
            NonNull::from(&mut id).cast(),
        )
    };
    if status != 0 || id == 0 {
        log::error!("audio System: default output device, OSStatus {status}");
        anyhow::bail!("재생 장치(스피커·헤드폰)를 찾을 수 없습니다. 출력 장치를 연결하세요.");
    }
    Ok(id)
}

fn device_uid(device: AudioObjectID) -> anyhow::Result<Retained<NSString>> {
    let mut uid: *mut c_void = std::ptr::null_mut();
    let mut size = size_of::<*mut c_void>() as u32;
    let property = address(kAudioDevicePropertyDeviceUID);
    let status = unsafe {
        AudioObjectGetPropertyData(
            device,
            NonNull::from(&property),
            0,
            std::ptr::null(),
            NonNull::from(&mut size),
            NonNull::from(&mut uid).cast(),
        )
    };
    if status != 0 || uid.is_null() {
        log::error!("audio System: device uid, OSStatus {status}");
        anyhow::bail!("재생 장치를 읽을 수 없습니다. 출력 장치 연결을 확인하세요.");
    }
    // The property is returned retained (create rule): take that reference.
    Ok(unsafe { Retained::from_raw(uid as *mut NSString) }.expect("non-null uid"))
}

fn device_nominal_rate(device: AudioObjectID) -> Option<f64> {
    let mut rate: f64 = 0.0;
    let mut size = size_of::<f64>() as u32;
    let property = address(kAudioDevicePropertyNominalSampleRate);
    let status = unsafe {
        AudioObjectGetPropertyData(
            device,
            NonNull::from(&property),
            0,
            std::ptr::null(),
            NonNull::from(&mut size),
            NonNull::from(&mut rate).cast(),
        )
    };
    (status == 0).then_some(rate)
}

/// False only when Core Audio says the device is dead; an unreadable property counts as alive.
fn device_alive(device: AudioObjectID) -> bool {
    let mut alive: u32 = 1;
    let mut size = size_of::<u32>() as u32;
    let property = address(kAudioDevicePropertyDeviceIsAlive);
    let status = unsafe {
        AudioObjectGetPropertyData(
            device,
            NonNull::from(&property),
            0,
            std::ptr::null(),
            NonNull::from(&mut size),
            NonNull::from(&mut alive).cast(),
        )
    };
    status != 0 || alive != 0
}

fn tap_format(tap: AudioObjectID) -> anyhow::Result<AudioStreamBasicDescription> {
    let mut format = MaybeUninit::<AudioStreamBasicDescription>::uninit();
    let mut size = size_of::<AudioStreamBasicDescription>() as u32;
    let property = address(kAudioTapPropertyFormat);
    let status = unsafe {
        AudioObjectGetPropertyData(
            tap,
            NonNull::from(&property),
            0,
            std::ptr::null(),
            NonNull::from(&mut size),
            NonNull::new(format.as_mut_ptr().cast::<c_void>()).expect("non-null"),
        )
    };
    check(status, "tap format")?;
    Ok(unsafe { format.assume_init() })
}

fn to_cfstring(cstr: &'static CStr) -> CFRetained<CFString> {
    unsafe {
        CFString::with_c_string(
            kCFAllocatorDefault,
            cstr.as_ptr(),
            0x0800_0100, /* UTF8 */
        )
    }
    .expect("cfstring")
}

/// The aggregate device description: one tap, private, tap starts with the device.
/// Rust form of the Objective-C dictionary in Apple's "Capturing system audio with Core Audio
/// taps" sample, as in cpal.
fn aggregate_properties(
    tap_uid: Retained<NSString>,
    agg_uid: &str,
    agg_name: &str,
) -> CFRetained<CFDictionary> {
    let tap_inner = unsafe {
        let dict = CFMutableDictionary::new(
            kCFAllocatorDefault,
            2,
            &kCFTypeDictionaryKeyCallBacks,
            &kCFTypeDictionaryValueCallBacks,
        )
        .expect("cfdictionary");
        CFMutableDictionary::set_value(
            Some(dict.as_ref()),
            &*to_cfstring(kAudioSubTapUIDKey) as *const _ as *const c_void,
            &*tap_uid as *const _ as *const c_void,
        );
        CFMutableDictionary::set_value(
            Some(dict.as_ref()),
            &*to_cfstring(kAudioSubTapDriftCompensationKey) as *const _ as *const c_void,
            &*NSNumber::initWithBool(NSNumber::alloc(), true) as *const _ as *const c_void,
        );
        dict
    };
    let tap_list = [tap_inner];
    let taps = unsafe {
        CFArray::new(
            kCFAllocatorDefault,
            tap_list.as_ptr() as *mut *const c_void,
            tap_list.len() as _,
            &kCFTypeArrayCallBacks,
        )
        .expect("cfarray")
    };
    unsafe {
        let dict = CFMutableDictionary::new(
            kCFAllocatorDefault,
            5,
            &kCFTypeDictionaryKeyCallBacks,
            &kCFTypeDictionaryValueCallBacks,
        )
        .expect("cfdictionary");
        CFMutableDictionary::set_value(
            Some(dict.as_ref()),
            &*to_cfstring(kAudioAggregateDeviceNameKey) as *const _ as *const c_void,
            &*CFString::from_str(agg_name) as *const _ as *const c_void,
        );
        CFMutableDictionary::set_value(
            Some(dict.as_ref()),
            &*to_cfstring(kAudioAggregateDeviceUIDKey) as *const _ as *const c_void,
            &*CFString::from_str(agg_uid) as *const _ as *const c_void,
        );
        CFMutableDictionary::set_value(
            Some(dict.as_ref()),
            &*to_cfstring(kAudioAggregateDeviceTapListKey) as *const _ as *const c_void,
            &*taps as *const _ as *const c_void,
        );
        CFMutableDictionary::set_value(
            Some(dict.as_ref()),
            &*to_cfstring(kAudioAggregateDeviceTapAutoStartKey) as *const _ as *const c_void,
            &*NSNumber::initWithBool(NSNumber::alloc(), true) as *const _ as *const c_void,
        );
        CFMutableDictionary::set_value(
            Some(dict.as_ref()),
            &*to_cfstring(kAudioEndPointDeviceIsPrivateKey) as *const _ as *const c_void,
            &*NSNumber::initWithBool(NSNumber::alloc(), true) as *const _ as *const c_void,
        );
        CFRetained::cast_unchecked::<CFDictionary>(dict)
    }
}
