extern crate winit;

use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};
use winit::dpi::LogicalSize;

use ash::{vk, ext};

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};

use anyhow::{anyhow, ensure, Context, Result};

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

const VALIDATION: ValidationInfo = ValidationInfo {
    enabled: true,
    required_validation_layers: [ "VK_LAYER_KHRONOS_validation" ],
};

struct QueueFamilyIndices {
    graphics_family: Option<u32>,
}

impl QueueFamilyIndices {
    pub fn is_complete(&self) -> bool {
        self.graphics_family.is_some()
    }
}

struct ValidationInfo {
    enabled: bool,
    required_validation_layers: [&'static str; 1],
}

unsafe extern "system" fn debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut c_void,
) -> vk::Bool32 {
    let severity = match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => "[Verbose]",
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => "[Warning]",
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => "[Error]",
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => "[Info]",
        _ => "[Unknown]",
    };
    let types = match message_type {
        vk::DebugUtilsMessageTypeFlagsEXT::GENERAL => "[General]",
        vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => "[Performance]",
        vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION => "[Validation]",
        _ => "[Unknown]",
    };
    let message = unsafe { CStr::from_ptr((*p_callback_data).p_message) }
        .to_str()
        .expect("Failed to convert debug message.")
        .to_owned();
    println!("[Debug]{}{}{:?}", severity, types, message);

    vk::FALSE
}

fn vk_to_string(raw_string_array: &[c_char]) -> String {
    let raw_string = unsafe {
        let pointer = raw_string_array.as_ptr();
        CStr::from_ptr(pointer)
    };

    raw_string
        .to_str()
        .expect("Failed to convert vulkan raw string.")
        .to_owned()
}


fn populate_debug_messenger_create_info(
    create_info: &mut vk::DebugUtilsMessengerCreateInfoEXT
) {
    create_info.s_type =
        vk::StructureType::DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    create_info.message_severity =
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
        // | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
        // | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
        | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR;
    create_info.message_type =
        vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION;
    create_info.pfn_user_callback = Some(debug_callback);
}

struct Engine {
    _entry: ash::Entry,
    instance: ash::Instance,
    debug_utils_instance: Option<ext::debug_utils::Instance>,
    debug_messenger: Option<vk::DebugUtilsMessengerEXT>,
    _physical_device: vk::PhysicalDevice,
}

impl Engine {
    pub fn new() -> Result<Self> {
        let entry = ash::Entry::linked();
        let instance = Self::create_instance(&entry)?;
        let (debug_utils_instance, debug_messenger) =
            Self::setup_debug_messenger(&entry, &instance)?;
        let physical_device = Self::pick_physical_device(&instance)?;

        Ok(Engine {
            _entry: entry,
            instance,
            debug_utils_instance,
            debug_messenger,
            _physical_device: physical_device,
        })
    }

    fn pick_physical_device(instance: &ash::Instance) -> Result<vk::PhysicalDevice> {
        let physical_devices = unsafe {
            instance
                .enumerate_physical_devices()
                .context("Failed to enumerate Physical Devices!")?
        };

        println!(
            "{} devices (GPU) found with vulkan support.",
            physical_devices.len()
        );

        let physical_device = physical_devices
            .iter()
            .find(|&device| { Self::is_device_suitable(&instance, *device) })
            .ok_or_else(|| anyhow!("Failed to find a suitable GPU!"))?;

        Ok(*physical_device)
    }

    fn create_instance(entry: &ash::Entry) -> Result<ash::Instance> {
        ensure!(
            !VALIDATION.enabled || Self::check_validation_layer_support(entry)?,
            "Validation layers requested, but not available!"
        );

        let app_name = CString::new("Hello Triangle")?;
        let engine_name = CString::new("No Engine")?;

        let app_info = vk::ApplicationInfo {
            s_type: vk::StructureType::APPLICATION_INFO,
            p_next: std::ptr::null(),
            p_application_name: app_name.as_ptr(),
            application_version: vk::make_api_version(0, 1, 0, 0),
            p_engine_name: engine_name.as_ptr(),
            engine_version: vk::make_api_version(0, 1, 0, 0),
            api_version: vk::make_api_version(0, 1, 0, 0),
            _marker: std::marker::PhantomData,
        };

        let mut required_extensions = vec![
            #[cfg(target_os = "macos")]
            vk::KHR_PORTABILITY_ENUMERATION_NAME.as_ptr(),
         ];

        if VALIDATION.enabled {
            required_extensions.push(vk::EXT_DEBUG_UTILS_NAME.as_ptr());
        }

        let validation_layers_raw: Vec<CString> = VALIDATION
            .required_validation_layers
            .iter()
            .map(|layer_name| CString::new(*layer_name).unwrap())
            .collect();
        let validation_layers: Vec<*const i8> = validation_layers_raw
            .iter()
            .map(|layer_name| layer_name.as_ptr())
            .collect();

        let mut create_info = vk::InstanceCreateInfo::default();
        create_info.s_type = vk::StructureType::INSTANCE_CREATE_INFO;
        create_info.p_application_info = &app_info;
        create_info.enabled_extension_count = required_extensions.len() as u32;
        create_info.pp_enabled_extension_names = required_extensions.as_ptr();
        #[cfg(target_os = "macos")] {
            create_info.flags = InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;
        }

        if VALIDATION.enabled {
            let mut debug_create_info =
                vk::DebugUtilsMessengerCreateInfoEXT::default();
            populate_debug_messenger_create_info(&mut debug_create_info);

            create_info.enabled_layer_count = validation_layers.len() as u32;
            create_info.pp_enabled_layer_names = validation_layers.as_ptr();
            create_info.p_next = &debug_create_info as
                *const vk::DebugUtilsMessengerCreateInfoEXT as *const c_void;
        }

        let instance = unsafe {
            entry
                .create_instance(&create_info, None)
                .context("Failed to create Vulkan instance")?
        };

        Ok(instance)
    }

    fn check_validation_layer_support(entry: &ash::Entry) -> Result<bool> {
        let layer_properties = unsafe {
            entry
                .enumerate_instance_layer_properties()
                .context("Failed to enumerate Instance Layers Properties!")?
        };

        if layer_properties.is_empty() {
            eprintln!("No available layers.");
            return Ok(false);
        } else {
            println!("Instance Available Layers: ");
            for layer in layer_properties.iter() {
                let layer_name = vk_to_string(&layer.layer_name);
                println!("\t{}", layer_name);
            }
        }

        'layer: for layer_name in VALIDATION.required_validation_layers.iter() {
            for layer_property in layer_properties.iter() {
                let test_layer_name = vk_to_string(&layer_property.layer_name);
                if (*layer_name) == test_layer_name {
                    continue 'layer;
                }
            }
            return Ok(false);
        }

        Ok(true)
    }

    fn setup_debug_messenger(entry: &ash::Entry, instance: &ash::Instance)
        -> Result<(Option<ext::debug_utils::Instance>, Option<vk::DebugUtilsMessengerEXT>)> {
        if !VALIDATION.enabled {
            return Ok((None, None));
        }

        let mut create_info = vk::DebugUtilsMessengerCreateInfoEXT::default();
        populate_debug_messenger_create_info(&mut create_info);

        let debug_utils_loader = ext::debug_utils::Instance::new(entry, instance);
        let debug_messenger = unsafe {
            debug_utils_loader
                .create_debug_utils_messenger(&create_info, None)
                .context("Failed to create Debug Utils Messenger!")?
        };

        Ok((Some(debug_utils_loader), Some(debug_messenger)))
    }

    fn is_device_suitable(instance: &ash::Instance, device: vk::PhysicalDevice) -> bool {
        let device_properties = unsafe { instance.get_physical_device_properties(device) };
        let device_name = vk_to_string(&device_properties.device_name);
        println!("\tDevice Name: {}", device_name);

        let indices = Self::find_queue_family(instance, device);
        indices.is_complete()
    }

    fn find_queue_family(instance: &ash::Instance, device: vk::PhysicalDevice) -> QueueFamilyIndices {
        let queue_families = unsafe { instance.get_physical_device_queue_family_properties(device) };

        let mut queue_family_indices = QueueFamilyIndices { graphics_family: None };

        for (index, queue_family) in queue_families.iter().enumerate() {
            if queue_family.queue_count > 0 && queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                queue_family_indices.graphics_family = Some(index as u32);
            }

            if queue_family_indices.is_complete() {
                break;
            }
        }

        queue_family_indices
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe {
            if VALIDATION.enabled {
                if let (Some(loader), Some(messenger)) = (&self.debug_utils_instance, &self.debug_messenger) {
                    loader.destroy_debug_utils_messenger(*messenger, None);
                }
            }
            self.instance.destroy_instance(None);
        }
    }
}

#[derive(Default)]
struct HelloTriangleApplication {
    window: Option<Window>,
    engine: Option<Engine>,
}

impl ApplicationHandler for HelloTriangleApplication {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window = event_loop
            .create_window(
                Window::default_attributes()
                    .with_title("Vulkan tutorial with Ash")
                    .with_inner_size(LogicalSize::new(WIDTH, HEIGHT)),
            )
            .unwrap();

        self.window = Some(window);
        self.engine = Some(Engine::new().expect("Failed to initialize Vulkan Engine!"));
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                println!("The close button was pressed; stopping");
                event_loop.exit();
            },
            WindowEvent::RedrawRequested => {
                self.window.as_ref().unwrap().request_redraw();
            }
            _ => (),
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();

    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.set_control_flow(ControlFlow::Wait);

    let mut app = HelloTriangleApplication::default();
    let _ = event_loop.run_app(&mut app);
}
