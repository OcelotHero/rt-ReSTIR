extern crate winit;

use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};
use winit::dpi::LogicalSize;

use ash::vk;
use ash::vk::InstanceCreateFlags;

use std::ptr;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

const VALIDATION: ValidationInfo = ValidationInfo {
    is_enabled: true,
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
    is_enabled: bool,
    required_validation_layers: [&'static str; 1],
}

#[derive(Default)]
struct HelloTriangleApplication {
    _entry: Option<ash::Entry>,
    window: Option<Window>,
    instance: Option<ash::Instance>,
    debug_messenger: Option<vk::DebugUtilsMessengerEXT>,
    debug_utils_instance: Option<ash::ext::debug_utils::Instance>,
    _physical_device: Option<vk::PhysicalDevice>,
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
        self.init_vulkan();
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
    *create_info = vk::DebugUtilsMessengerCreateInfoEXT {
        s_type: vk::StructureType::DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        p_next: ptr::null(),
        flags: vk::DebugUtilsMessengerCreateFlagsEXT::empty(),
        message_severity: vk::DebugUtilsMessageSeverityFlagsEXT::WARNING |
            // vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE |
            // vk::DebugUtilsMessageSeverityFlagsEXT::INFO |
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
        message_type: vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
            | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
            | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
        pfn_user_callback: Some(debug_callback),
        p_user_data: ptr::null_mut(),
        _marker: std::marker::PhantomData,
    }
}

impl HelloTriangleApplication {
    fn check_validation_layer_support(&mut self) -> bool {
        // if support validation layer, then return true

        let layer_properties = unsafe {
            self._entry
                .as_ref()
                .unwrap()
                .enumerate_instance_layer_properties()
            }.expect("Failed to enumerate Instance Layers Properties!");

        if layer_properties.len() <= 0 {
            eprintln!("No available layers.");
            return false;
        } else {
            println!("Instance Available Layers: ");
            for layer in layer_properties.iter() {
                let layer_name = vk_to_string(&layer.layer_name);
                println!("\t{}", layer_name);
            }
        }

        for required_layer_name in VALIDATION.required_validation_layers.iter() {
            let mut is_layer_found = false;

            for layer_property in layer_properties.iter() {
                let test_layer_name = vk_to_string(&layer_property.layer_name);
                if (*required_layer_name) == test_layer_name {
                    is_layer_found = true;
                    break;
                }
            }

            if is_layer_found == false {
                return false;
            }
        }

        true
    }

    fn init_vulkan(&mut self) {
        self.create_instance();
        self.setup_debug_messenger();
        self.pick_physical_device();
    }

    fn create_instance(&mut self) {
        let entry = ash::Entry::linked();
        self._entry = Some(entry);

        if VALIDATION.is_enabled && self.check_validation_layer_support() == false {
            panic!("Validation layers requested, but not available!");
        }

        let app_name = CString::new("Hello Triangle").unwrap();
        let engine_name = CString::new("No Engine").unwrap();

        let app_info = ash::vk::ApplicationInfo {
            s_type: ash::vk::StructureType::APPLICATION_INFO,
            p_next: std::ptr::null(),
            p_application_name: app_name.as_ptr(),
            application_version: ash::vk::make_api_version(0, 1, 0, 0),
            p_engine_name: engine_name.as_ptr(),
            engine_version: ash::vk::make_api_version(0, 1, 0, 0),
            api_version: ash::vk::make_api_version(0, 1, 0, 0),
            _marker: std::marker::PhantomData,
        };

        let mut required_extensions = if cfg!(target_os = "macos") {
            vec![
                // ash::vk::KHR_SURFACE_EXTENSION_NAME.as_ptr(),
                // ash::vk::KHR_MACOS_SURFACE_EXTENSION_NAME.as_ptr(),
                ash::vk::KHR_PORTABILITY_ENUMERATION_NAME.as_ptr(),
            ]
        } else {
            vec![]
        };
        if VALIDATION.is_enabled {
            required_extensions.push(ash::vk::EXT_DEBUG_UTILS_NAME.as_ptr());
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
        create_info._marker = std::marker::PhantomData;

        if cfg!(target_os = "macos") {
            create_info.flags = InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;
        }
        if VALIDATION.is_enabled {
            let mut debug_create_info =
                vk::DebugUtilsMessengerCreateInfoEXT::default();
            populate_debug_messenger_create_info(&mut debug_create_info);

            create_info.enabled_layer_count = validation_layers.len() as u32;
            create_info.pp_enabled_layer_names = validation_layers.as_ptr();
            create_info.p_next = &debug_create_info as
                *const vk::DebugUtilsMessengerCreateInfoEXT as *const c_void;
        }

        let instance = unsafe {
            self._entry
                .as_ref()
                .unwrap()
                .create_instance(&create_info, None)
                .expect("Failed to create Vulkan instance")
        };

        self.instance = Some(instance);
    }

    fn setup_debug_messenger(&mut self) {
        if !VALIDATION.is_enabled {
            return;
        }

        let mut create_info = vk::DebugUtilsMessengerCreateInfoEXT::default();
        populate_debug_messenger_create_info(&mut create_info);

        let debug_utils_loader = ash::ext::debug_utils::Instance::new(
            self._entry.as_ref().unwrap(),
            self.instance.as_ref().unwrap(),
        );
        let debug_messenger = unsafe {
            debug_utils_loader
                .create_debug_utils_messenger(&create_info, None)
                .expect("Failed to create Debug Utils Messenger!")
        };

        self.debug_utils_instance = Some(debug_utils_loader);
        self.debug_messenger = Some(debug_messenger);
    }

    fn pick_physical_device(&mut self) {
        let physical_devices = unsafe {
            self.instance
                .as_ref()
                .unwrap()
                .enumerate_physical_devices()
                .expect("Failed to enumerate Physical Devices!")
        };

        println!(
            "{} devices (GPU) found with vulkan support.",
            physical_devices.len()
        );

        let physical_device = physical_devices
            .iter()
            .find(|&device| { self.is_device_suitable(device) });

        match physical_device {
            Some(device) => self._physical_device = Some(*device),
            None => panic!("Failed to find a suitable GPU!"),
        }
    }

    fn is_device_suitable(&self, device: &vk::PhysicalDevice) -> bool {
        let device_properties = unsafe {
            self.instance
                .as_ref()
                .unwrap()
                .get_physical_device_properties(*device)
        };
        let device_name = vk_to_string(&device_properties.device_name);
        println!("\tDevice Name: {}", device_name);

        let indices = self.find_queue_family(*device);
        indices.is_complete()
    }

    fn find_queue_family(&self, device: vk::PhysicalDevice) -> QueueFamilyIndices {
        let queue_families = unsafe {
            self.instance
                .as_ref()
                .unwrap()
                .get_physical_device_queue_family_properties(device)
        };

        let mut queue_family_indices = QueueFamilyIndices {
            graphics_family: None,
        };

        let mut index = 0;
        for queue_family in queue_families.iter() {
            if queue_family.queue_count > 0
                && queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
            {
                queue_family_indices.graphics_family = Some(index);
            }

            if queue_family_indices.is_complete() {
                break;
            }

            index += 1;
        }

        queue_family_indices
    }

    fn main_loop(&mut self, _event_loop: &ActiveEventLoop) {
    }
}

impl Drop for HelloTriangleApplication {
    fn drop(&mut self) {
        unsafe {
            if VALIDATION.is_enabled {
                self.debug_utils_instance
                    .as_ref()
                    .unwrap()
                    .destroy_debug_utils_messenger(
                        self.debug_messenger.unwrap(),
                        None,
                    );
            }
            self.instance.as_ref().unwrap().destroy_instance(None);
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
