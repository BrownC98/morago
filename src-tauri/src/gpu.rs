//! GPU use for the local models: translation (llama.cpp) and Fun-ASR
//! (transcribe.cpp). Both run on the one ggml of
//! llama.cpp, so they share its GPU backend, the pack and this placement.
//! - Windows: CUDA on NVIDIA GPUs. The CUDA runtime DLLs are not in the
//!   installer: the "GPU 구성 요소" pack downloads them from NVIDIA on demand
//!   (`download_pack`). build.rs delay-loads them, so the app starts and runs
//!   on the CPU without them. The first load comes before `main` (nvcc kernel
//!   registration in static constructors), so the delay-load hook finds the
//!   pack by its fixed folder (`app_dir`). A pack installed while the app runs
//!   takes effect after a restart.
//! - macOS: Metal. Linux (development only): CPU.
//!
//! On CUDA a failed GPU allocation can abort the process (ggml `CUDA_CHECK`).
//! `place` does not check the free GPU memory: the user accepts that risk.

use crate::models::{FileSpec, ModelKind, Spec};
use crate::types::{DownloadProgress, GpuSetting, ModelDevice};
use anyhow::{Context, anyhow, bail};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::{LlamaBackendDevice, LlamaBackendDeviceType, list_llama_ggml_backend_devices};
use serde::Serialize;
use std::ffi::{CStr, c_char, c_void};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

/// Download id of the pack (the `download_model` command) and its folder.
pub const PACK_ID: &str = "cuda-12.9.1";

/// NVIDIA's CUDA 12.9.1 redistributable archives (redistrib_12.9.1.json).
/// License: CUDA EULA, Attachment A lists cudart and cuBLAS as redistributable.
const PACK: Spec = Spec {
    id: PACK_ID,
    // download_spec reads only `id` and `files`.
    kind: ModelKind::Translation,
    name: "GPU 구성 요소",
    note: "",
    files: &[
        FileSpec {
            urls: &[
                "https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/windows-x86_64/cuda_cudart-windows-x86_64-12.9.79-archive.zip",
            ],
            name: "cudart.zip",
            size: 3_521_238,
            sha256: "179e9c43b0735ffe67207b3da556eb5a0c50f3047961882b7657d3b822d34ef8",
        },
        FileSpec {
            urls: &[
                "https://developer.download.nvidia.com/compute/cuda/redist/libcublas/windows-x86_64/libcublas-windows-x86_64-12.9.1.4-archive.zip",
            ],
            name: "cublas.zip",
            size: 549_755_186,
            sha256: "d534d98b0b453a98914dbf3adf47d7e84b55037abf02f87466439e1dcef581ed",
        },
    ],
};

/// One file that the pack takes out of an archive.
struct PackFile {
    archive: &'static str,
    entry: &'static str,
    name: &'static str,
    size: u64,
}

const PACK_FILES: &[PackFile] = &[
    PackFile {
        archive: "cudart.zip",
        entry: "cuda_cudart-windows-x86_64-12.9.79-archive/bin/cudart64_12.dll",
        name: "cudart64_12.dll",
        size: 583_680,
    },
    PackFile {
        archive: "cudart.zip",
        entry: "cuda_cudart-windows-x86_64-12.9.79-archive/LICENSE",
        name: "LICENSE-cudart.txt",
        size: 63_522,
    },
    PackFile {
        archive: "cublas.zip",
        entry: "libcublas-windows-x86_64-12.9.1.4-archive/bin/cublas64_12.dll",
        name: "cublas64_12.dll",
        size: 102_518_272,
    },
    PackFile {
        archive: "cublas.zip",
        entry: "libcublas-windows-x86_64-12.9.1.4-archive/bin/cublasLt64_12.dll",
        name: "cublasLt64_12.dll",
        size: 668_669_952,
    },
    PackFile {
        archive: "cublas.zip",
        entry: "libcublas-windows-x86_64-12.9.1.4-archive/LICENSE",
        name: "LICENSE-cublas.txt",
        size: 67_876,
    },
];

/// Checked values: CUDA 12.9.1 release notes, tables 2 and 3 (minor version
/// compatibility; CUDA 12.8 GA is the first toolkit for Blackwell).
const DRIVER_NOTE: &str = "NVIDIA 드라이버 528.33 이상(RTX 50 시리즈는 570.65 이상)이 필요합니다.";
pub const CHOOSE_OTHER: &str = "설정 › GPU 가속에서 다른 GPU나 「끔」을 고른 뒤 다시 시작하세요.";

/// The app data folder `gpu`, as Tauri `app_data_dir` gives it: the roaming
/// data folder and the identifier of tauri.conf.json.
pub fn app_dir() -> Option<PathBuf> {
    dirs::data_dir().map(|d| d.join("com.morago.app").join("gpu"))
}

/// Folder of the pack DLLs, under the app data folder `gpu`.
fn pack_bin(gpu_dir: &Path) -> PathBuf {
    gpu_dir.join(PACK_ID).join("bin")
}

fn pack_installed(gpu_dir: &Path) -> bool {
    let bin = pack_bin(gpu_dir);
    PACK_FILES.iter().all(|f| {
        std::fs::metadata(bin.join(f.name)).is_ok_and(|m| m.is_file() && m.len() == f.size)
    })
}

static GPU_DIR: OnceLock<PathBuf> = OnceLock::new();
/// The delay-load hook loaded the CUDA runtime from the pack.
static PACK_ACTIVE: AtomicBool = AtomicBool::new(false);

/// At app start: the folder for `status` and `place`. It must be `app_dir()`.
pub fn init(gpu_dir: &Path) {
    let _ = GPU_DIR.set(gpu_dir.to_path_buf());
    #[cfg(windows)]
    std::hint::black_box(delayload::keep());
}

/// Downloads the NVIDIA archives (pinned SHA-256), takes the DLLs out and
/// deletes the archives. Progress events use `PACK_ID` as the model id.
pub fn download_pack(
    gpu_dir: &Path,
    progress: &dyn Fn(DownloadProgress),
    cancel: &AtomicBool,
) -> anyhow::Result<()> {
    if pack_installed(gpu_dir) {
        return Ok(());
    }
    // One cuBLAS archive is 549 MB: a dropped connection is not rare (host, 2026-11-08).
    let mut attempt = 1;
    loop {
        match crate::models::download_spec(gpu_dir, &PACK, progress, cancel) {
            Err(e) if attempt < 3 && !cancel.load(Ordering::Relaxed) => {
                log::warn!("GPU pack download attempt {attempt} failed: {e:#}");
                attempt += 1;
            }
            result => break result?,
        }
    }
    let archives = gpu_dir.join(PACK_ID);
    let part = archives.join("bin.part");
    let _ = std::fs::remove_dir_all(&part);
    std::fs::create_dir_all(&part).context("GPU 구성 요소 폴더를 만들 수 없습니다")?;
    if let Err(e) = unpack(&archives, &part, PACK_FILES, cancel) {
        let _ = std::fs::remove_dir_all(&part);
        return Err(e);
    }
    let bin = pack_bin(gpu_dir);
    if bin.exists() {
        std::fs::remove_dir_all(&bin).context("이전 GPU 구성 요소를 지울 수 없습니다")?;
    }
    std::fs::rename(&part, &bin).context("GPU 구성 요소의 이름을 바꿀 수 없습니다")?;
    for f in PACK.files {
        let _ = std::fs::remove_file(archives.join(f.name));
    }
    Ok(())
}

/// Copies `files` out of the archives in `archives` into `out`. The zip
/// reader checks each CRC-32; the size is checked here.
fn unpack(
    archives: &Path,
    out: &Path,
    files: &[PackFile],
    cancel: &AtomicBool,
) -> anyhow::Result<()> {
    for f in files {
        if cancel.load(Ordering::Relaxed) {
            bail!("다운로드를 취소했습니다.");
        }
        let fail = |e: &dyn std::fmt::Display| {
            anyhow!(
                "GPU 구성 요소를 풀지 못했습니다({}): {e}. 저장 공간을 확인하고 다시 내려받으세요.",
                f.name
            )
        };
        let zip = std::fs::File::open(archives.join(f.archive)).map_err(|e| fail(&e))?;
        let mut zip = zip::ZipArchive::new(std::io::BufReader::new(zip)).map_err(|e| fail(&e))?;
        let mut src = zip.by_name(f.entry).map_err(|e| fail(&e))?;
        let mut dst = std::fs::File::create(out.join(f.name)).map_err(|e| fail(&e))?;
        let n = std::io::copy(&mut src, &mut dst).map_err(|e| fail(&e))?;
        if n != f.size {
            return Err(fail(&format!("크기 {n}, 기대 {}", f.size)));
        }
    }
    Ok(())
}

/// llama.cpp allows one backend per process. The log of llama.cpp,
/// transcribe.cpp and ggml goes to `log`; the last error lines give the cause
/// of a failure (`failure_cause`). Call it before the first Fun-ASR model.
pub fn llama_backend() -> anyhow::Result<&'static LlamaBackend> {
    static BACKEND: OnceLock<Result<LlamaBackend, String>> = OnceLock::new();
    BACKEND
        .get_or_init(|| {
            let backend = LlamaBackend::init().map_err(|e| e.to_string())?;
            // SAFETY: the callbacks are valid for the life of the process.
            // transcribe_log_set also sets the ggml log; llama_log_set comes
            // after it and sets the ggml log to `on_log`.
            unsafe {
                transcribe_cpp::sys::transcribe_log_set(
                    Some(on_transcribe_log),
                    std::ptr::null_mut(),
                );
                llama_cpp_sys_2::llama_log_set(Some(on_log), std::ptr::null_mut());
            }
            Ok(backend)
        })
        .as_ref()
        .map_err(|e| anyhow!("로컬 모델 엔진을 시작하지 못했습니다: {e}"))
}

static LLAMA_ERRORS: Mutex<Vec<String>> = Mutex::new(Vec::new());

unsafe extern "C" fn on_log(
    level: llama_cpp_sys_2::ggml_log_level,
    text: *const c_char,
    _: *mut c_void,
) {
    let level = match level {
        llama_cpp_sys_2::GGML_LOG_LEVEL_ERROR => log::Level::Error,
        llama_cpp_sys_2::GGML_LOG_LEVEL_WARN => log::Level::Warn,
        _ => return,
    };
    // SAFETY: llama.cpp passes a NUL-terminated string.
    unsafe { keep_log(level, text) }
}

unsafe extern "C" fn on_transcribe_log(
    level: transcribe_cpp::sys::transcribe_log_level,
    text: *const c_char,
    _: *mut c_void,
) {
    use transcribe_cpp::sys::transcribe_log_level as L;
    let level = match level {
        L::TRANSCRIBE_LOG_LEVEL_ERROR => log::Level::Error,
        L::TRANSCRIBE_LOG_LEVEL_WARN => log::Level::Warn,
        _ => return,
    };
    // SAFETY: transcribe.cpp passes a NUL-terminated string.
    unsafe { keep_log(level, text) }
}

/// SAFETY: `text` is null or a NUL-terminated string.
unsafe fn keep_log(level: log::Level, text: *const c_char) {
    if text.is_null() {
        return;
    }
    // SAFETY: see the function.
    let text = unsafe { CStr::from_ptr(text) }.to_string_lossy();
    let text = text.trim();
    log::log!(level, "ggml: {text}");
    if level == log::Level::Error
        && let Ok(mut errors) = LLAMA_ERRORS.lock()
    {
        if errors.len() >= 8 {
            errors.remove(0);
        }
        errors.push(text.to_string());
    }
}

/// Forgets earlier llama.cpp errors, before a call whose cause we may need.
pub fn clear_llama_errors() {
    if let Ok(mut errors) = LLAMA_ERRORS.lock() {
        errors.clear();
    }
}

/// The cause of a llama.cpp failure for the user: its last error lines, or
/// the error of the call.
pub fn failure_cause(e: &dyn std::fmt::Display) -> String {
    let log = LLAMA_ERRORS
        .lock()
        .map(|mut errors| std::mem::take(&mut *errors))
        .unwrap_or_default();
    if log.iter().any(|l| l.contains("out of memory")) {
        "GPU 메모리 부족".into()
    } else if let Some(last) = log.last() {
        last.clone()
    } else {
        e.to_string()
    }
}

/// A GPU error in a running session. The engine stops and saves the session.
#[derive(Debug)]
pub struct GpuFailure(pub String);

impl std::fmt::Display for GpuFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for GpuFailure {}

/// The error when the translation model does not load on the GPU (D2 B: the
/// meeting does not start).
pub fn load_error(gpu_name: &str, cause: &str) -> anyhow::Error {
    model_load_error("번역 모델", gpu_name, cause)
}

/// `model`: "번역 모델", "음성 인식 모델".
pub fn model_load_error(model: &str, gpu_name: &str, cause: &str) -> anyhow::Error {
    let cause = cause.trim_end_matches('.');
    anyhow!("{model}을 {gpu_name}에 불러오지 못했습니다. 원인: {cause}. {CHOOSE_OTHER}")
}

/// A GPU that the setting list offers.
struct Gpu {
    name: String,
    backend: String,
    /// Why the app cannot use it, for the user.
    problem: Option<String>,
}

fn gpus() -> &'static [Gpu] {
    static GPUS: OnceLock<Vec<Gpu>> = OnceLock::new();
    GPUS.get_or_init(detect)
}

/// NVIDIA GPUs from the driver: the list does not need the pack.
#[cfg(windows)]
fn detect() -> Vec<Gpu> {
    let Some(driver) = win::probe_nvidia() else {
        return Vec::new();
    };
    driver
        .cards
        .into_iter()
        .map(|card| {
            // RTX 50 (compute 12.x) needs a CUDA 12.8 driver, the others 12.0.
            let need = if card.compute.0 >= 12 { 12_080 } else { 12_000 };
            let problem = if card.compute < (7, 5) {
                Some(format!(
                    "이 GPU는 지원하지 않습니다: {}. GPU 가속에는 RTX 20 시리즈 이후의 NVIDIA GPU가 필요합니다.",
                    card.name
                ))
            } else if driver.cuda_version < need {
                Some(format!("NVIDIA 드라이버가 오래되었습니다. {DRIVER_NOTE}"))
            } else {
                None
            };
            Gpu {
                name: card.name,
                backend: "CUDA".into(),
                problem,
            }
        })
        .collect()
}

#[cfg(not(windows))]
fn detect() -> Vec<Gpu> {
    if llama_backend().is_err() {
        return Vec::new();
    }
    list_llama_ggml_backend_devices()
        .into_iter()
        .filter(|d| matches!(d.device_type, LlamaBackendDeviceType::Gpu))
        .map(|d| Gpu {
            name: d.description,
            backend: if d.backend == "MTL" {
                "Metal".into()
            } else {
                d.backend
            },
            problem: None,
        })
        .collect()
}

/// The ggml device of a GPU, with its free memory now.
fn ggml_device(name: &str) -> Option<LlamaBackendDevice> {
    llama_backend().ok()?;
    list_llama_ggml_backend_devices()
        .into_iter()
        .find(|d| matches!(d.device_type, LlamaBackendDeviceType::Gpu) && d.description == name)
}

/// Free memory of a ggml device now, in bytes.
#[cfg(test)]
pub fn free_memory(index: usize) -> Option<u64> {
    list_llama_ggml_backend_devices()
        .into_iter()
        .find(|d| d.index == index)
        .map(|d| d.memory_free as u64)
}

/// Runs `f` and gives the lowest free memory of GPU `index` in MiB while it
/// runs (read every 20 ms), for the memory tests with real models.
#[cfg(test)]
pub fn lowest_free_while<T>(index: Option<usize>, f: impl FnOnce() -> T) -> (T, Option<u64>) {
    let Some(index) = index else {
        return (f(), None);
    };
    let done = AtomicBool::new(false);
    let lowest = std::sync::atomic::AtomicU64::new(u64::MAX);
    let out = std::thread::scope(|s| {
        s.spawn(|| {
            while !done.load(Ordering::Relaxed) {
                if let Some(free) = free_memory(index) {
                    lowest.fetch_min(free >> 20, Ordering::Relaxed);
                }
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
        });
        let out = f();
        done.store(true, Ordering::Relaxed);
        out
    });
    (out, Some(lowest.load(Ordering::Relaxed)))
}

const MIB: u64 = 1024 * 1024;

fn gb(bytes: u64) -> String {
    format!("{:.1} GB", bytes as f64 / (1024 * MIB) as f64)
}

/// Where a local model goes: a ggml device, or the CPU.
#[derive(Debug, Clone)]
pub struct Placement {
    /// ggml device index and GPU name. `None` = CPU.
    pub gpu: Option<(usize, String)>,
    pub status: ModelDevice,
}

impl Placement {
    pub fn cpu(detail: impl Into<String>) -> Self {
        Self {
            gpu: None,
            status: ModelDevice {
                gpu: false,
                detail: detail.into(),
            },
        }
    }
}

/// Picks the device of a local model (translation, Fun-ASR): the
/// chosen GPU, else the CPU with the reason. No memory check: a CUDA
/// out-of-memory error ends the process. An error (the chosen GPU is not on
/// this PC) stops the meeting start (D2 B).
pub fn place(setting: &GpuSetting) -> anyhow::Result<Placement> {
    let gpu = match setting {
        GpuSetting::Off => return Ok(Placement::cpu("GPU 가속을 껐습니다.")),
        GpuSetting::Auto => match gpus()
            .iter()
            .find(|g| g.problem.is_none())
            .or(gpus().first())
        {
            Some(g) => g,
            None => return Ok(Placement::cpu(no_gpu_text())),
        },
        GpuSetting::Device { name } => gpus()
            .iter()
            .find(|g| g.name == *name)
            .ok_or_else(|| anyhow!("선택한 GPU를 찾을 수 없습니다: {name}. {CHOOSE_OTHER}"))?,
    };
    if let Some(problem) = &gpu.problem {
        return Ok(Placement::cpu(problem.clone()));
    }
    if cfg!(windows) && !PACK_ACTIVE.load(Ordering::Relaxed) {
        let installed = GPU_DIR.get().is_some_and(|d| pack_installed(d));
        return Ok(Placement::cpu(if installed {
            format!("앱을 다시 시작하면 {}를 씁니다.", gpu.name)
        } else {
            format!("GPU 구성 요소를 내려받으면 {}를 씁니다.", gpu.name)
        }));
    }
    let Some(dev) = ggml_device(&gpu.name) else {
        return Ok(Placement::cpu(format!(
            "{}를 쓸 수 없습니다: {}가 이 GPU를 찾지 못했습니다.",
            gpu.name, gpu.backend
        )));
    };
    Ok(Placement {
        gpu: Some((dev.index, gpu.name.clone())),
        status: ModelDevice {
            gpu: true,
            detail: format!("{} · {}", gpu.name, gpu.backend),
        },
    })
}

fn no_gpu_text() -> &'static str {
    if cfg!(windows) {
        "NVIDIA GPU가 없습니다."
    } else {
        "쓸 수 있는 GPU가 없습니다."
    }
}

/// The "GPU 구성 요소" pack for the settings screen.
#[derive(Debug, Serialize)]
pub struct PackStatus {
    pub id: &'static str,
    pub size_bytes: u64,
    pub installed: bool,
    /// Installed while the app runs: a restart uses it.
    pub needs_restart: bool,
    pub note: String,
}

/// Settings › GPU 가속.
#[derive(Debug, Serialize)]
pub struct GpuStatus {
    /// GPU names for the setting list.
    pub gpus: Vec<String>,
    pub hint: Option<&'static str>,
    /// Windows with a usable NVIDIA GPU only.
    pub pack: Option<PackStatus>,
    /// Device of the local translation model; `None` when a service translates.
    pub translate: Option<ModelDevice>,
    /// Device of the local speech recognition model; `None` when a service recognizes.
    pub asr: Option<ModelDevice>,
}

/// `translate`, `asr`: devices of the local models (`engine::model_devices`).
pub fn status(translate: Option<ModelDevice>, asr: Option<ModelDevice>) -> GpuStatus {
    let gpus = gpus();
    let pack = (cfg!(windows) && gpus.iter().any(|g| g.problem.is_none())).then(|| {
        let installed = GPU_DIR.get().is_some_and(|d| pack_installed(d));
        PackStatus {
            id: PACK_ID,
            size_bytes: PACK.files.iter().map(|f| f.size).sum(),
            installed,
            needs_restart: installed && !PACK_ACTIVE.load(Ordering::Relaxed),
            note: format!(
                "NVIDIA GPU에서 음성 인식과 번역을 돌리는 CUDA 런타임입니다. NVIDIA 서버에서 내려받고, 풀면 {}입니다. {DRIVER_NOTE}",
                gb(PACK_FILES.iter().map(|f| f.size).sum())
            ),
        }
    });
    GpuStatus {
        gpus: gpus.iter().map(|g| g.name.clone()).collect(),
        hint: cfg!(windows).then_some("GPU 가속은 NVIDIA GPU(CUDA)만 씁니다."),
        pack,
        translate,
        asr,
    }
}

#[cfg(windows)]
mod win {
    //! Win32 and NVIDIA driver calls. kernel32 is linked by default.
    use std::ffi::{CStr, c_char, c_void};
    use std::os::windows::ffi::OsStrExt;
    use std::path::Path;

    unsafe extern "system" {
        fn LoadLibraryW(name: *const u16) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
        fn LoadLibraryExW(name: *const u16, file: *mut c_void, flags: u32) -> *mut c_void;
    }

    fn wide(s: &std::ffi::OsStr) -> Vec<u16> {
        s.encode_wide().chain(std::iter::once(0)).collect()
    }

    /// Loads a DLL by its full path. Its own imports (cublasLt for cuBLAS)
    /// come from the same folder (LOAD_WITH_ALTERED_SEARCH_PATH).
    pub fn load_library(path: &Path) -> Option<*mut c_void> {
        const LOAD_WITH_ALTERED_SEARCH_PATH: u32 = 0x8;
        let path = wide(path.as_os_str());
        // SAFETY: a NUL-terminated wide string; no file handle.
        let module = unsafe {
            LoadLibraryExW(
                path.as_ptr(),
                std::ptr::null_mut(),
                LOAD_WITH_ALTERED_SEARCH_PATH,
            )
        };
        (!module.is_null()).then_some(module)
    }

    pub struct Card {
        pub name: String,
        /// Compute capability (major, minor).
        pub compute: (i32, i32),
    }

    pub struct Driver {
        /// CUDA version of the driver, for example 12090 for 12.9.
        pub cuda_version: i32,
        pub cards: Vec<Card>,
    }

    /// The NVIDIA driver API (nvcuda.dll, in System32 with the driver). `None`:
    /// no NVIDIA driver or no NVIDIA GPU.
    pub fn probe_nvidia() -> Option<Driver> {
        type Init = unsafe extern "system" fn(u32) -> i32;
        type GetInt = unsafe extern "system" fn(*mut i32) -> i32;
        type DeviceGet = unsafe extern "system" fn(*mut i32, i32) -> i32;
        type DeviceName = unsafe extern "system" fn(*mut c_char, i32, i32) -> i32;
        type DeviceAttribute = unsafe extern "system" fn(*mut i32, i32, i32) -> i32;
        const COMPUTE_MAJOR: i32 = 75;
        const COMPUTE_MINOR: i32 = 76;

        let name = wide("nvcuda.dll".as_ref());
        // SAFETY: a NUL-terminated wide string.
        let lib = unsafe { LoadLibraryW(name.as_ptr()) };
        if lib.is_null() {
            return None;
        }
        let symbol = |name: &CStr| {
            // SAFETY: `lib` is a loaded module; `name` is NUL-terminated.
            let f = unsafe { GetProcAddress(lib, name.as_ptr()) };
            (!f.is_null()).then_some(f)
        };
        // SAFETY: the types are the documented signatures of these exports.
        let (init, driver_version, device_count, device_get, device_name, device_attribute) = unsafe {
            (
                std::mem::transmute::<*mut c_void, Init>(symbol(c"cuInit")?),
                std::mem::transmute::<*mut c_void, GetInt>(symbol(c"cuDriverGetVersion")?),
                std::mem::transmute::<*mut c_void, GetInt>(symbol(c"cuDeviceGetCount")?),
                std::mem::transmute::<*mut c_void, DeviceGet>(symbol(c"cuDeviceGet")?),
                std::mem::transmute::<*mut c_void, DeviceName>(symbol(c"cuDeviceGetName")?),
                std::mem::transmute::<*mut c_void, DeviceAttribute>(symbol(
                    c"cuDeviceGetAttribute",
                )?),
            )
        };
        let (mut version, mut count) = (0, 0);
        // SAFETY: valid out pointers; the driver API wants cuInit first.
        unsafe {
            if init(0) != 0 || driver_version(&mut version) != 0 || device_count(&mut count) != 0 {
                return None;
            }
        }
        let mut cards = Vec::new();
        for ordinal in 0..count {
            let mut dev = 0;
            let mut buf = [0 as c_char; 256];
            let (mut major, mut minor) = (0, 0);
            // SAFETY: valid out pointers and buffer length.
            let ok = unsafe {
                device_get(&mut dev, ordinal) == 0
                    && device_name(buf.as_mut_ptr(), buf.len() as i32, dev) == 0
                    && device_attribute(&mut major, COMPUTE_MAJOR, dev) == 0
                    && device_attribute(&mut minor, COMPUTE_MINOR, dev) == 0
            };
            if ok {
                // SAFETY: the driver writes a NUL-terminated name into `buf`.
                let name = unsafe { CStr::from_ptr(buf.as_ptr()) };
                cards.push(Card {
                    name: name.to_string_lossy().trim().to_string(),
                    compute: (major, minor),
                });
            }
        }
        Some(Driver {
            cuda_version: version,
            cards,
        })
    }
}

#[cfg(windows)]
mod delayload {
    //! Hooks of the MSVC delay-load helper (build.rs `/DELAYLOAD`).
    //! - Notify hook: the CUDA runtime and cuBLAS come only from the pack
    //!   folder, never from PATH. Without the pack it gives the exe module.
    //! - Failure hook: without it a missing DLL or function raises an
    //!   exception at the first call and the process ends. The hook gives the
    //!   exe module (the load "succeeds") and, for each function, a stub that
    //!   returns error 100 (cudaErrorNoDevice, CUDA_ERROR_NO_DEVICE). ggml then
    //!   finds no CUDA device and llama.cpp uses the CPU. Proven in R1 (t1).
    use std::ffi::{CStr, c_char, c_void};
    use std::sync::atomic::Ordering;

    #[repr(C)]
    struct DelayLoadInfo {
        cb: u32,
        pidd: *const c_void,
        ppfn: *mut *const c_void,
        dll: *const c_char,
        by_name: i32,
        name: *const c_char,
        hmod_cur: isize,
        pfn_cur: *const c_void,
        last_error: u32,
    }

    type Hook = unsafe extern "system" fn(u32, *const DelayLoadInfo) -> *const c_void;

    unsafe extern "system" {
        fn GetModuleHandleW(name: *const u16) -> isize;
    }

    const PRE_LOAD_LIB: u32 = 1;
    const FAIL_LOAD_LIB: u32 = 3;
    const FAIL_GET_PROC: u32 = 4;

    fn exe_module() -> *const c_void {
        // SAFETY: a null name gives the exe module.
        unsafe { GetModuleHandleW(std::ptr::null()) as *const c_void }
    }

    /// A pack DLL from the pack folder, else the exe module (stubs).
    fn from_pack(dll: &str) -> *const c_void {
        let module = super::app_dir()
            .filter(|d| super::pack_installed(d))
            .and_then(|d| super::win::load_library(&super::pack_bin(&d).join(dll)));
        match module {
            Some(m) => {
                super::PACK_ACTIVE.store(true, Ordering::Relaxed);
                m
            }
            None => exe_module(),
        }
    }

    unsafe extern "system" fn notify_hook(
        notify: u32,
        info: *const DelayLoadInfo,
    ) -> *const c_void {
        if notify != PRE_LOAD_LIB {
            return std::ptr::null();
        }
        // SAFETY: the delay-load helper passes a valid info block with a
        // NUL-terminated DLL name.
        let dll = unsafe { CStr::from_ptr((*info).dll) }.to_string_lossy();
        match dll.to_ascii_lowercase().as_str() {
            "cudart64_12.dll" | "cublas64_12.dll" => from_pack(&dll),
            // nvcuda.dll: the driver, from System32.
            _ => std::ptr::null(),
        }
    }

    unsafe extern "system" fn no_device() -> i32 {
        100
    }

    unsafe extern "system" fn error_string(_: i32) -> *const c_char {
        c"CUDA runtime not installed".as_ptr()
    }

    unsafe extern "system" fn hook(notify: u32, info: *const DelayLoadInfo) -> *const c_void {
        // SAFETY: the delay-load helper passes a valid info block.
        let info = unsafe { &*info };
        match notify {
            FAIL_LOAD_LIB => exe_module(),
            FAIL_GET_PROC => {
                let name = if info.by_name != 0 && !info.name.is_null() {
                    // SAFETY: the import name is NUL-terminated.
                    unsafe { CStr::from_ptr(info.name) }.to_bytes()
                } else {
                    b""
                };
                match name {
                    b"cudaGetErrorString" | b"cudaGetErrorName" => error_string as *const c_void,
                    _ => no_device as *const c_void,
                }
            }
            _ => std::ptr::null(),
        }
    }

    /// The helper in delayimp.lib reads these symbols.
    #[unsafe(no_mangle)]
    #[allow(non_upper_case_globals)]
    static __pfnDliNotifyHook2: Hook = notify_hook;
    #[unsafe(no_mangle)]
    #[allow(non_upper_case_globals)]
    static __pfnDliFailureHook2: Hook = hook;

    /// A reference that keeps the hooks in the link.
    pub fn keep() -> [*const c_void; 2] {
        [
            (&raw const __pfnDliNotifyHook2).cast(),
            (&raw const __pfnDliFailureHook2).cast(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn off_is_the_cpu_without_a_gpu_call() {
        let p = place(&GpuSetting::Off).unwrap();
        assert!(p.gpu.is_none());
        assert!(!p.status.gpu);
        assert_eq!(p.status.detail, "GPU 가속을 껐습니다.");
    }

    #[test]
    fn a_missing_chosen_gpu_stops_the_start() {
        let err = place(&GpuSetting::Device {
            name: "No Such GPU 9000".into(),
        })
        .unwrap_err()
        .to_string();
        assert!(err.contains("No Such GPU 9000"), "{err}");
        assert!(err.contains("「끔」"), "{err}");
    }

    #[test]
    fn sizes_show_in_gb() {
        assert_eq!(gb(3 * 1024 * MIB / 2), "1.5 GB");
    }

    #[test]
    fn pack_takes_files_from_both_archives() {
        assert_eq!(PACK.files.len(), 2);
        for f in PACK.files {
            assert!(f.urls[0].starts_with("https://developer.download.nvidia.com/"));
            assert!(PACK_FILES.iter().any(|p| p.archive == f.name));
        }
        for dll in ["cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll"] {
            assert!(PACK_FILES.iter().any(|p| p.name == dll), "{dll}");
        }
    }

    fn write_zip(path: &Path, entries: &[(&str, &[u8])]) {
        let mut zip = zip::ZipWriter::new(std::fs::File::create(path).unwrap());
        for (name, data) in entries {
            zip.start_file(*name, zip::write::SimpleFileOptions::default())
                .unwrap();
            zip.write_all(data).unwrap();
        }
        zip.finish().unwrap();
    }

    #[test]
    fn unpack_copies_and_checks_sizes() {
        let dir = tempfile::tempdir().unwrap();
        write_zip(
            &dir.path().join("a.zip"),
            &[("x/bin/a.dll", b"abc"), ("x/LICENSE", b"license")],
        );
        let out = dir.path().join("out");
        std::fs::create_dir_all(&out).unwrap();
        let files = [
            PackFile {
                archive: "a.zip",
                entry: "x/bin/a.dll",
                name: "a.dll",
                size: 3,
            },
            PackFile {
                archive: "a.zip",
                entry: "x/LICENSE",
                name: "LICENSE.txt",
                size: 7,
            },
        ];
        unpack(dir.path(), &out, &files, &AtomicBool::new(false)).unwrap();
        assert_eq!(std::fs::read(out.join("a.dll")).unwrap(), b"abc");

        let wrong = [PackFile {
            archive: "a.zip",
            entry: "x/bin/a.dll",
            name: "a.dll",
            size: 4,
        }];
        let err = unpack(dir.path(), &out, &wrong, &AtomicBool::new(false)).unwrap_err();
        assert!(err.to_string().contains("풀지 못했습니다"), "{err}");

        let cancelled = unpack(dir.path(), &out, &files, &AtomicBool::new(true)).unwrap_err();
        assert!(cancelled.to_string().contains("취소"), "{cancelled}");
    }

    #[test]
    fn a_damaged_pack_is_not_installed() {
        let dir = tempfile::tempdir().unwrap();
        assert!(!pack_installed(dir.path()));
        let bin = pack_bin(dir.path());
        std::fs::create_dir_all(&bin).unwrap();
        for f in PACK_FILES {
            std::fs::write(bin.join(f.name), b"short").unwrap();
        }
        assert!(!pack_installed(dir.path()));
    }

    #[test]
    fn app_dir_is_the_tauri_app_data_folder() {
        let conf: serde_json::Value =
            serde_json::from_str(include_str!("../tauri.conf.json")).unwrap();
        let dir = app_dir().unwrap();
        assert_eq!(dir.file_name().unwrap(), "gpu");
        assert_eq!(
            dir.parent().unwrap(),
            dirs::data_dir()
                .unwrap()
                .join(conf["identifier"].as_str().unwrap())
        );
    }

    /// Run: `MORAGO_TEST_GPU_DIR=<dir> cargo test --lib gpu::tests::pack_downloads -- --ignored --nocapture`
    #[test]
    #[ignore = "downloads 528 MB from NVIDIA"]
    fn pack_downloads_and_installs() {
        let dir = std::env::var_os("MORAGO_TEST_GPU_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| std::env::temp_dir().join("morago-gpu-test"));
        let before = place(&GpuSetting::Auto).map(|p| p.status);
        println!("before: {before:?}");
        let last = Mutex::new(0u64);
        download_pack(
            &dir,
            &|p| *last.lock().unwrap() = p.done_bytes,
            &AtomicBool::new(false),
        )
        .unwrap();
        assert!(pack_installed(&dir));
        assert!(!dir.join(PACK_ID).join("cudart.zip").exists());
        println!(
            "downloaded {} bytes into {}",
            last.lock().unwrap(),
            dir.display()
        );
    }

    /// The test exe links like the app exe (build.rs link args): the CUDA
    /// DLLs must be delay-load imports only, or Windows does not start the
    /// app on a PC without them.
    #[cfg(windows)]
    #[test]
    fn cuda_dlls_are_delay_loaded() {
        let exe = std::fs::read(std::env::current_exe().unwrap()).unwrap();
        let (imports, delayed) = pe_imports(&exe);
        for dll in ["nvcuda.dll", "cudart64_12.dll", "cublas64_12.dll"] {
            assert!(
                !imports.iter().any(|i| i.eq_ignore_ascii_case(dll)),
                "{dll} is a static import: {imports:?}"
            );
            assert!(
                delayed.iter().any(|i| i.eq_ignore_ascii_case(dll)),
                "{dll} is not delay-loaded: {delayed:?}"
            );
        }
    }

    /// DLL names of the import and delay-import tables of a PE32+ file.
    #[cfg(windows)]
    fn pe_imports(pe: &[u8]) -> (Vec<String>, Vec<String>) {
        let u16_at = |o: usize| u16::from_le_bytes([pe[o], pe[o + 1]]) as usize;
        let u32_at = |o: usize| u32::from_le_bytes(pe[o..o + 4].try_into().unwrap()) as usize;
        let nt = u32_at(0x3c);
        let sections = u16_at(nt + 6);
        let opt = nt + 24;
        let table = opt + u16_at(nt + 20);
        let offset = |rva: usize| {
            (0..sections).find_map(|i| {
                let s = table + i * 40;
                let (size, va, raw) = (u32_at(s + 8), u32_at(s + 12), u32_at(s + 20));
                (va..va + size).contains(&rva).then(|| rva - va + raw)
            })
        };
        let name_at = |rva: usize| {
            let o = offset(rva).unwrap();
            let end = pe[o..].iter().position(|&b| b == 0).unwrap();
            String::from_utf8_lossy(&pe[o..o + end]).into_owned()
        };
        // Data directories of PE32+: 1 = imports (20-byte entries, name at 12),
        // 13 = delay imports (32-byte entries, name at 4).
        let list = |dir: usize, entry: usize, name: usize| {
            let rva = u32_at(opt + 112 + dir * 8);
            let mut out = Vec::new();
            if rva == 0 {
                return out;
            }
            let mut o = offset(rva).unwrap();
            while u32_at(o + name) != 0 {
                out.push(name_at(u32_at(o + name)));
                o += entry;
            }
            out
        };
        (list(1, 20, 12), list(13, 32, 4))
    }
}
