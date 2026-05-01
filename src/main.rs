extern crate winit;

use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::dpi::LogicalSize;

use ash::{vk, ext, khr};

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};

use anyhow::{anyhow, ensure, Context, Result};

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

const VALIDATION: ValidationInfo = ValidationInfo {
    enabled: true,
    required_validation_layers: [ "VK_LAYER_KHRONOS_validation" ],
};

#[derive(Default)]
struct QueueFamilyIndices {
    graphics_family: Option<u32>,
    present_family: Option<u32>,
}

impl QueueFamilyIndices {
    pub fn is_complete(&self) -> bool {
        self.graphics_family.is_some() && self.present_family.is_some()
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
    surface_instance: khr::surface::Instance,
    surface: vk::SurfaceKHR,
    debug_utils_instance: Option<ext::debug_utils::Instance>,
    debug_messenger: Option<vk::DebugUtilsMessengerEXT>,

    _physical_device: vk::PhysicalDevice,
    device: ash::Device,

    _graphics_queue: vk::Queue,
    _present_queue: vk::Queue,
}

impl Engine {
    pub fn new(window: &Window, required_extensions: &[*const i8]) -> Result<Self> {
        let entry = ash::Entry::linked();
        let instance = Self::create_instance(&entry, required_extensions)?;
        let (surface_instance, surface) = Self::create_surface(&entry, &instance, window)?;
        let (debug_utils_instance, debug_messenger) =
            Self::setup_debug_messenger(&entry, &instance)?;
        let physical_device = Self::pick_physical_device(&instance, &surface_instance, &surface)?;
        let (device, family_indices) =
            Self::create_logical_device(&instance, physical_device, &surface_instance, &surface)?;
        let graphics_queue =
            unsafe { device.get_device_queue(family_indices.graphics_family.context("Graphics")?, 0) };
        let present_queue =
            unsafe { device.get_device_queue(family_indices.present_family.context("Present")?, 0) };

        Ok(Engine {
            _entry: entry,
            instance,
            debug_utils_instance,
            debug_messenger,
            surface_instance,
            surface,
            _physical_device: physical_device,
            device,
            _graphics_queue: graphics_queue,
            _present_queue: present_queue,
        })
    }

    fn create_instance(entry: &ash::Entry, required_extensions: &[*const i8])
        -> Result<ash::Instance> {
        ensure!(
            !VALIDATION.enabled || Self::check_validation_layer_support(entry)?,
            "Validation layers requested, but not available!"
        );

        let app_name = CString::new("Hello Triangle")?;
        let engine_name = CString::new("No Engine")?;
        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(vk::make_api_version(0, 1, 0, 0))
            .engine_name(&engine_name)
            .engine_version(vk::make_api_version(0, 1, 0, 0))
            .api_version(vk::make_api_version(0, 1, 0, 0));

        let mut required_extensions = required_extensions.to_vec();
        #[cfg(target_os = "macos")]
        {
            required_extensions.push(vk::KHR_PORTABILITY_ENUMERATION_NAME.as_ptr());
        }
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

        let mut create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&required_extensions);

        #[cfg(target_os = "macos")] {
            create_info = create_info
                .flags(vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR);
        }

        let mut debug_create_info =
            vk::DebugUtilsMessengerCreateInfoEXT::default();
        populate_debug_messenger_create_info(&mut debug_create_info);
        if VALIDATION.enabled {
            create_info = create_info
                .enabled_layer_names(&validation_layers)
                .push_next(&mut debug_create_info);
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

        let debug_utils_instance = ext::debug_utils::Instance::new(entry, instance);
        let debug_messenger = unsafe {
            debug_utils_instance
                .create_debug_utils_messenger(&create_info, None)
                .context("Failed to create Debug Utils Messenger!")?
        };

        Ok((Some(debug_utils_instance), Some(debug_messenger)))
    }

    fn is_device_suitable(
        instance: &ash::Instance,
        device: vk::PhysicalDevice,
        surface_instance: &khr::surface::Instance,
        surface: &vk::SurfaceKHR
    ) -> bool {
        let device_properties = unsafe { instance.get_physical_device_properties(device) };
        let device_name = vk_to_string(&device_properties.device_name);
        println!("\tDevice Name: {}", device_name);

        let indices = Self::find_queue_family(instance, device, surface_instance, surface);
        indices.is_complete()
    }

    fn find_queue_family(
        instance: &ash::Instance,
        device: vk::PhysicalDevice,
        surface_instance: &khr::surface::Instance,
        surface: &vk::SurfaceKHR
    ) -> QueueFamilyIndices {
        let queue_families = unsafe { instance.get_physical_device_queue_family_properties(device) };

        let mut queue_family_indices = QueueFamilyIndices::default();

        for (index, queue_family) in queue_families.iter().enumerate() {
            if queue_family.queue_count > 0 && queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                queue_family_indices.graphics_family = Some(index as u32);
            }

            let present_support = unsafe {
                surface_instance.get_physical_device_surface_support(device, index as u32, *surface)
            };
            if queue_family.queue_count > 0 && present_support.unwrap_or(false) {
                queue_family_indices.present_family = Some(index as u32);
            }

            if queue_family_indices.is_complete() {
                break;
            }
        }

        queue_family_indices
    }

    fn pick_physical_device(
        instance: &ash::Instance,
        surface_instance: &khr::surface::Instance,
        surface: &vk::SurfaceKHR
    ) -> Result<vk::PhysicalDevice> {
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
            .find(|&device| { Self::is_device_suitable(&instance, *device, surface_instance, surface) })
            .ok_or_else(|| anyhow!("Failed to find a suitable GPU!"))?;

        Ok(*physical_device)
    }

    fn create_logical_device(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        surface_instance: &khr::surface::Instance,
        surface: &vk::SurfaceKHR
    ) -> Result<(ash::Device, QueueFamilyIndices)> {
        let indices = Self::find_queue_family(instance, physical_device, surface_instance, surface);

        let unique_queue_families = [indices.graphics_family, indices.present_family]
            .iter()
            .filter_map(|&family| family)
            .collect::<std::collections::HashSet<u32>>();

        let queue_priority = [1.0_f32];
        let queue_create_info = unique_queue_families
            .iter()
            .map(|&family| {
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(family)
                    .queue_priorities(&queue_priority)
            })
            .collect::<Vec<_>>();

        let device_features = vk::PhysicalDeviceFeatures::default();

        let mut create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_info)
            .enabled_features(&device_features);

        let validation_layers_raw: Vec<CString> = VALIDATION
            .required_validation_layers
            .iter()
            .map(|layer_name| CString::new(*layer_name).unwrap())
            .collect();
        let validation_layers: Vec<*const i8> = validation_layers_raw
            .iter()
            .map(|layer_name| layer_name.as_ptr())
            .collect();

        if VALIDATION.enabled {
            create_info = create_info.enabled_layer_names(&validation_layers);
        }

        let device = unsafe {
            instance
                .create_device(physical_device, &create_info, None)
                .context("Failed to create Logical Device!")?
        };

        Ok((device, indices))
    }

    fn create_surface(entry: &ash::Entry, instance: &ash::Instance, window: &Window)
        -> Result<(khr::surface::Instance, vk::SurfaceKHR)> {
        let surface = unsafe {
            ash_window::create_surface(
                entry,
                instance,
                window.display_handle()?.as_raw(),
                window.window_handle()?.as_raw(),
                None
            )
                .context("Failed to create Surface!")?
        };
        let surface_instance = khr::surface::Instance::new(entry, instance);

        Ok((surface_instance, surface))
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
            self.surface_instance.destroy_surface(self.surface, None);
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
        let required_extensions =
            ash_window::enumerate_required_extensions(window.display_handle().unwrap().as_raw())
                .expect("Failed to enumerate required extensions for surface creation!");

        self.engine = Some(Engine::new(&window, &required_extensions).expect("Failed to initialize Vulkan Engine!"));
        self.window = Some(window);
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
