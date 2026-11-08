//! GPU use for the local translation model (llama.cpp).
//! - Windows: CUDA on NVIDIA GPUs. The CUDA runtime DLLs are not in the
//!   installer: the "GPU 구성 요소" pack downloads them from NVIDIA on demand
//!   (`download_pack`). build.rs delay-loads them, so the app starts and runs
//!   on the CPU without them. A pack installed while the app runs takes effect
//!   after a restart: ggml finds its CUDA devices once per process.
//! - macOS: Metal. Linux (development only): CPU.
//!
//! On CUDA a failed GPU allocation can abort the process (ggml `CUDA_CHECK`),
//! so `place` checks the free GPU memory before it picks a GPU.

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
/// The loader finds the pack DLLs (set once at start).
static PACK_ACTIVE: AtomicBool = AtomicBool::new(false);

/// At app start, before any llama.cpp call: makes the pack DLLs visible to
/// the loader when the pack is installed.
pub fn init(gpu_dir: &Path) {
    let _ = GPU_DIR.set(gpu_dir.to_path_buf());
    #[cfg(windows)]
    {
        std::hint::black_box(delayload::keep());
        if pack_installed(gpu_dir) {
            if win::set_dll_directory(&pack_bin(gpu_dir)) {
                PACK_ACTIVE.store(true, Ordering::Relaxed);
            } else {
                log::warn!("SetDllDirectoryW failed; GPU pack not used");
            }
        }
    }
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
    crate::models::download_spec(gpu_dir, &PACK, progress, cancel)?;
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

/// llama.cpp allows one backend per process. Its log goes to `log`; the last
/// error lines give the cause of a failure (`failure_cause`).
pub fn llama_backend() -> anyhow::Result<&'static LlamaBackend> {
    static BACKEND: OnceLock<Result<LlamaBackend, String>> = OnceLock::new();
    BACKEND
        .get_or_init(|| {
            let backend = LlamaBackend::init().map_err(|e| e.to_string())?;
            // SAFETY: `on_log` is a valid callback for the life of the process.
            unsafe { llama_cpp_sys_2::llama_log_set(Some(on_log), std::ptr::null_mut()) };
            Ok(backend)
        })
        .as_ref()
        .map_err(|e| anyhow!("번역 엔진을 시작하지 못했습니다: {e}"))
}

static LLAMA_ERRORS: Mutex<Vec<String>> = Mutex::new(Vec::new());

unsafe extern "C" fn on_log(
    level: llama_cpp_sys_2::ggml_log_level,
    text: *const c_char,
    _: *mut c_void,
) {
    if text.is_null() {
        return;
    }
    // SAFETY: llama.cpp passes a NUL-terminated string.
    let text = unsafe { CStr::from_ptr(text) }.to_string_lossy();
    let text = text.trim();
    if level == llama_cpp_sys_2::GGML_LOG_LEVEL_ERROR {
        log::error!("llama.cpp: {text}");
        if let Ok(mut errors) = LLAMA_ERRORS.lock() {
            if errors.len() >= 8 {
                errors.remove(0);
            }
            errors.push(text.to_string());
        }
    } else if level == llama_cpp_sys_2::GGML_LOG_LEVEL_WARN {
        log::warn!("llama.cpp: {text}");
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

/// The error when a model does not load on the GPU (D2 B: the meeting does not start).
pub fn load_error(gpu_name: &str, cause: &str) -> anyhow::Error {
    let cause = cause.trim_end_matches('.');
    anyhow!("번역 모델을 {gpu_name}에 불러오지 못했습니다. 원인: {cause}. {CHOOSE_OTHER}")
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
pub fn free_memory(index: usize) -> Option<u64> {
    list_llama_ggml_backend_devices()
        .into_iter()
        .find(|d| d.index == index)
        .map(|d| d.memory_free as u64)
}

/// GPU memory that a model of `model_bytes` needs: the weights, the small
/// translation context and the CUDA buffers, with a margin.
fn needed_memory(model_bytes: u64) -> u64 {
    model_bytes + model_bytes * 3 / 10 + 512 * MIB
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

/// Picks the device for a model of `model_bytes`. An error (the chosen GPU is
/// not on this PC) stops the meeting start (D2 B); every other reason not to
/// use a GPU gives the CPU and says why.
pub fn place(setting: &GpuSetting, model_bytes: u64) -> anyhow::Result<Placement> {
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
    let need = needed_memory(model_bytes);
    let free = dev.memory_free as u64;
    if free < need {
        return Ok(Placement::cpu(format!(
            "GPU 메모리가 모자랍니다: {}에 남은 메모리 {}, 필요 {}.",
            gpu.name,
            gb(free),
            gb(need)
        )));
    }
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
}

/// `translate_bytes`: size of the local translation model, if one is set.
pub fn status(setting: &GpuSetting, translate_bytes: Option<u64>) -> GpuStatus {
    let gpus = gpus();
    let pack = (cfg!(windows) && gpus.iter().any(|g| g.problem.is_none())).then(|| {
        let installed = GPU_DIR.get().is_some_and(|d| pack_installed(d));
        PackStatus {
            id: PACK_ID,
            size_bytes: PACK.files.iter().map(|f| f.size).sum(),
            installed,
            needs_restart: installed && !PACK_ACTIVE.load(Ordering::Relaxed),
            note: format!(
                "NVIDIA GPU에서 번역하는 CUDA 런타임입니다. NVIDIA 서버에서 내려받고, 풀면 {}입니다. {DRIVER_NOTE}",
                gb(PACK_FILES.iter().map(|f| f.size).sum())
            ),
        }
    });
    GpuStatus {
        gpus: gpus.iter().map(|g| g.name.clone()).collect(),
        hint: cfg!(windows).then_some("GPU 가속은 NVIDIA GPU(CUDA)만 씁니다."),
        pack,
        translate: translate_bytes.map(|bytes| match place(setting, bytes) {
            Ok(p) => p.status,
            Err(e) => ModelDevice {
                gpu: false,
                detail: e.to_string(),
            },
        }),
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
        fn SetDllDirectoryW(path: *const u16) -> i32;
    }

    fn wide(s: &std::ffi::OsStr) -> Vec<u16> {
        s.encode_wide().chain(std::iter::once(0)).collect()
    }

    /// Adds `dir` to the DLL search path of the process (before System32 and PATH).
    pub fn set_dll_directory(dir: &Path) -> bool {
        let dir = wide(dir.as_os_str());
        // SAFETY: a NUL-terminated wide string.
        unsafe { SetDllDirectoryW(dir.as_ptr()) != 0 }
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
    //! Failure hook of the MSVC delay-load helper (build.rs `/DELAYLOAD`).
    //! Without it a missing CUDA DLL raises an exception at the first call and
    //! the process ends. The hook gives the helper the exe module (the load
    //! "succeeds") and, for each function, a stub that returns error 100
    //! (cudaErrorNoDevice, CUDA_ERROR_NO_DEVICE). ggml then finds no CUDA
    //! device and llama.cpp uses the CPU. Proven in R1 (test crate t1).
    use std::ffi::{CStr, c_char, c_void};

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

    const FAIL_LOAD_LIB: u32 = 3;
    const FAIL_GET_PROC: u32 = 4;

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
            // SAFETY: a null name gives the exe module.
            FAIL_LOAD_LIB => unsafe { GetModuleHandleW(std::ptr::null()) as *const c_void },
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

    /// The helper in delayimp.lib reads this symbol.
    #[unsafe(no_mangle)]
    #[allow(non_upper_case_globals)]
    static __pfnDliFailureHook2: Hook = hook;

    /// A reference that keeps the hook in the link.
    pub fn keep() -> *const Hook {
        &raw const __pfnDliFailureHook2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn off_is_the_cpu_without_a_gpu_call() {
        let p = place(&GpuSetting::Off, 1 << 30).unwrap();
        assert!(p.gpu.is_none());
        assert!(!p.status.gpu);
        assert_eq!(p.status.detail, "GPU 가속을 껐습니다.");
    }

    #[test]
    fn a_missing_chosen_gpu_stops_the_start() {
        let err = place(
            &GpuSetting::Device {
                name: "No Such GPU 9000".into(),
            },
            1 << 30,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("No Such GPU 9000"), "{err}");
        assert!(err.contains("「끔」"), "{err}");
    }

    #[test]
    fn memory_need_has_a_margin() {
        // Gemma 4 E2B fits an 8 GB card; 12B does not.
        let e2b = 3_349_516_256;
        let b12 = 6_975_879_296;
        assert!(needed_memory(e2b) < 6 * 1024 * MIB);
        assert!(needed_memory(b12) > 8 * 1024 * MIB);
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
