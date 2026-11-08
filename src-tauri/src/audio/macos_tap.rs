//! macOS system audio through a Core Audio process tap (macOS 14.6+).
//!
//! Same method as cpal's loopback (cpal 0.18, Apache-2.0 / MIT, `coreaudio/macos/loopback.rs`):
//! a tap on the default output device, read through a private aggregate device. cpal builds it
//! only when the output device has no input channels. A full-duplex device (AirPods, a USB
//! headset, an audio interface) would give the microphone instead, so this module builds the
//! tap itself and reads it with an IOProc. Needs NSAudioCaptureUsageDescription (Info.plist).
//!
//! Not testable without a Mac. CI compiles it; see the manual test steps in the task result.

use super::worker::Feeder;
use super::{Event, Running};
use crossbeam_channel::Sender;
use objc2::{AnyThread, rc::Retained};
use objc2_core_audio::{
    AudioDeviceCreateIOProcID, AudioDeviceDestroyIOProcID, AudioDeviceIOProcID, AudioDeviceStart,
    AudioDeviceStop, AudioHardwareCreateAggregateDevice, AudioHardwareCreateProcessTap,
    AudioHardwareDestroyAggregateDevice, AudioHardwareDestroyProcessTap,
    AudioObjectGetPropertyData, AudioObjectID, AudioObjectPropertyAddress, CATapDescription,
    CATapMuteBehavior, kAudioAggregateDeviceNameKey, kAudioAggregateDeviceTapAutoStartKey,
    kAudioAggregateDeviceTapListKey, kAudioAggregateDeviceUIDKey, kAudioDevicePropertyDeviceUID,
    kAudioEndPointDeviceIsPrivateKey, kAudioHardwarePropertyDefaultOutputDevice,
    kAudioObjectPropertyElementMain, kAudioObjectPropertyScopeGlobal, kAudioObjectSystemObject,
    kAudioSubTapDriftCompensationKey, kAudioSubTapUIDKey, kAudioTapPropertyFormat,
};
use objc2_core_audio_types::{
    AudioBufferList, AudioStreamBasicDescription, AudioTimeStamp, kAudioFormatLinearPCM,
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

/// `kAudioFormatFlagIsFloat` and `kAudioFormatFlagIsNonInterleaved` (CoreAudioBaseTypes.h).
const FLAG_FLOAT: u32 = 1;
const FLAG_NON_INTERLEAVED: u32 = 1 << 5;

/// What the IOProc needs. Lives in a `Box` that `TapStream` frees after the IOProc is gone.
struct IoContext {
    feeder: Feeder,
    rate: u32,
    non_interleaved: bool,
    #[allow(dead_code)]
    events: Sender<Event>,
}

/// A running tap. Dropping it stops the IOProc and destroys the aggregate device and the tap.
pub struct TapStream {
    tap_id: AudioObjectID,
    aggregate_id: Option<AudioObjectID>,
    proc_id: AudioDeviceIOProcID,
    started: bool,
    context: *mut IoContext,
    /// The default output device that the tap listens to.
    output_device: AudioObjectID,
}

// The raw pointer is owned by this value and used by the IOProc only until `drop` stops it.
unsafe impl Send for TapStream {}

impl Running for TapStream {
    /// The default output changed (for example AirPods connected): the tap still listens to
    /// the old device and would give silence.
    fn stale(&self) -> bool {
        default_output_device().is_ok_and(|id| id != self.output_device)
    }
}

impl Drop for TapStream {
    fn drop(&mut self) {
        unsafe {
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
        }
    }
}

pub fn open(feeder: Feeder, events: Sender<Event>, _generation: u64) -> anyhow::Result<TapStream> {
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
        output_device,
    };

    // 2. The format of the tap (rate and layout).
    let format = tap_format(tap_id)?;
    if format.mFormatID != kAudioFormatLinearPCM
        || format.mFormatFlags & FLAG_FLOAT == 0
        || format.mBitsPerChannel != 32
    {
        log::error!("audio System: unexpected tap format {format:?}");
        anyhow::bail!("시스템 소리의 형식을 읽을 수 없습니다.");
    }
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

    // 4. Read it with an IOProc.
    let context = Box::into_raw(Box::new(IoContext {
        feeder,
        rate: format.mSampleRate as u32,
        non_interleaved: format.mFormatFlags & FLAG_NON_INTERLEAVED != 0,
        events,
    }));
    stream.context = context;
    let mut proc_id: AudioDeviceIOProcID = None;
    let status = unsafe {
        AudioDeviceCreateIOProcID(
            aggregate_id,
            Some(io_proc),
            context.cast::<c_void>(),
            NonNull::from(&mut proc_id),
        )
    };
    check(status, "IOProc")?;
    stream.proc_id = proc_id;
    check(unsafe { AudioDeviceStart(aggregate_id, proc_id) }, "start")?;
    stream.started = true;
    Ok(stream)
}

/// Called by Core Audio on its IO thread: queue the mono block, nothing else.
unsafe extern "C-unwind" fn io_proc(
    _device: AudioObjectID,
    _now: NonNull<AudioTimeStamp>,
    input: NonNull<AudioBufferList>,
    _input_time: NonNull<AudioTimeStamp>,
    _output: NonNull<AudioBufferList>,
    _output_time: NonNull<AudioTimeStamp>,
    client: *mut c_void,
) -> OSStatus {
    // SAFETY: `client` is the `IoContext` box of the `TapStream`, freed only after the IOProc is
    // stopped and destroyed. Core Audio gives a valid buffer list for the call.
    let ctx = unsafe { &*(client as *const IoContext) };
    let list = unsafe { input.as_ref() };
    let count = list.mNumberBuffers as usize;
    let buffers = unsafe { std::slice::from_raw_parts(list.mBuffers.as_ptr(), count) };
    let frames = if ctx.non_interleaved {
        buffers
            .iter()
            .map(|b| b.mDataByteSize as usize / 4)
            .min()
            .unwrap_or(0)
    } else {
        buffers.first().map_or(0, |b| {
            b.mDataByteSize as usize / 4 / (b.mNumberChannels.max(1) as usize)
        })
    };
    if frames == 0 {
        return 0;
    }
    let block = Duration::from_secs_f64(frames as f64 / f64::from(ctx.rate));
    let now = Instant::now();
    let start = now.checked_sub(block).unwrap_or(now);
    ctx.feeder.feed(start, ctx.rate, |mono| {
        if ctx.non_interleaved {
            mono.resize(frames, 0.0);
            for b in buffers {
                if b.mData.is_null() {
                    continue;
                }
                let data = unsafe { std::slice::from_raw_parts(b.mData as *const f32, frames) };
                for (m, s) in mono.iter_mut().zip(data) {
                    *m += *s;
                }
            }
            let scale = 1.0 / buffers.len().max(1) as f32;
            for m in mono.iter_mut() {
                *m *= scale;
            }
        } else if let Some(b) = buffers.first() {
            if b.mData.is_null() {
                return;
            }
            let channels = b.mNumberChannels.max(1) as usize;
            let data =
                unsafe { std::slice::from_raw_parts(b.mData as *const f32, frames * channels) };
            super::pipeline::downmix_into(mono, data, channels, |s| s);
        }
    });
    0
}

/// Korean message for a Core Audio call that failed. The most likely cause is the permission.
fn check(status: OSStatus, what: &str) -> anyhow::Result<()> {
    if status == 0 {
        return Ok(());
    }
    log::error!("audio System: {what} failed, OSStatus {status}");
    anyhow::bail!(
        "시스템 소리를 시작하지 못했습니다. 시스템 설정 > 개인정보 보호 및 보안 > 화면 및 시스템 오디오 녹음에서 Morago를 켜고 앱을 다시 시작하세요."
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
