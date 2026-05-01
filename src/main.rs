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

use anyhow::{bail, ensure, Context, Result};

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

const VALIDATION: ValidationInfo = ValidationInfo {
    enabled: true,
    required_validation_layers: [ "VK_LAYER_KHRONOS_validation" ],
};
const DEVICE_EXTENSIONS: [&'static str; 1] = [ "VK_KHR_swapchain" ];

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

struct SwapchainSupportDetails {
    capabilities: vk::SurfaceCapabilitiesKHR,
    formats: Vec<vk::SurfaceFormatKHR>,
    present_modes: Vec<vk::PresentModeKHR>,
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

fn vk_to_string(raw_string_array: &[c_char]) -> Result<String> {
    let raw_string = unsafe {
        let pointer = raw_string_array.as_ptr();
        CStr::from_ptr(pointer)
    };

    Ok(raw_string
        .to_str()
        .context("Failed to convert vulkan raw string.")?
        .to_owned())
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

    swapchain_instance: khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
    _swapchain_format: vk::Format,
    _swapchain_extent: vk::Extent2D,
    _swapchain_images: Vec<vk::Image>,
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
        let (swapchain_instance, swapchain, image_format, extent, images) =
            Self::create_swapchain(&instance, &device, physical_device, &surface_instance, &surface)?;

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
            swapchain_instance,
            swapchain,
            _swapchain_format: image_format,
            _swapchain_extent: extent,
            _swapchain_images: images,
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
                let layer_name = vk_to_string(&layer.layer_name)?;
                println!("\t{}", layer_name);
            }
        }

        'layer: for layer_name in VALIDATION.required_validation_layers.iter() {
            for layer_property in layer_properties.iter() {
                let test_layer_name = vk_to_string(&layer_property.layer_name)?;
                if (*layer_name) == test_layer_name {
                    continue 'layer;
                }
            }
            return Ok(false);
        }

        Ok(true)
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

        let physical_device = 'selection: {
            for device in physical_devices.iter() {
                if Self::is_device_suitable(instance, *device, surface_instance, surface)? {
                    break 'selection device;
                }
            }
            bail!("Failed to find a suitable GPU!")
        };

        Ok(*physical_device)
    }

    fn is_device_suitable(
        instance: &ash::Instance,
        device: vk::PhysicalDevice,
        surface_instance: &khr::surface::Instance,
        surface: &vk::SurfaceKHR
    ) -> Result<bool> {
        let device_properties = unsafe { instance.get_physical_device_properties(device) };
        let device_name = vk_to_string(&device_properties.device_name)?;
        println!("\tDevice Name: {}", device_name);

        let indices = Self::find_queue_family(instance, device, surface_instance, surface)?;
        let extensions_supported = Self::check_device_extension_support(instance, device)?;
        let swapchain_adequate = extensions_supported && {
            let details = Self::query_swapchain_support(device, surface_instance, surface)?;
            !details.formats.is_empty() && !details.present_modes.is_empty()
        };

        Ok(indices.is_complete() && extensions_supported && swapchain_adequate)
    }

    fn find_queue_family(
        instance: &ash::Instance,
        device: vk::PhysicalDevice,
        surface_instance: &khr::surface::Instance,
        surface: &vk::SurfaceKHR
    ) -> Result<QueueFamilyIndices> {
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

        Ok(queue_family_indices)
    }

    fn check_device_extension_support(instance: &ash::Instance, device: vk::PhysicalDevice)
        -> Result<bool> {
        let extension_properties = unsafe {
            instance
                .enumerate_device_extension_properties(device)
                .context("Failed to enumerate Device Extension Properties!")?
        };

        if extension_properties.is_empty() {
            eprintln!("No available device extensions.");
            return Ok(false);
        } else {
            println!("Available Device Extensions: ");
            for extension in extension_properties.iter() {
                let extension_name = vk_to_string(&extension.extension_name)?;
                println!("\t\tName: {}, Version: {}", extension_name, extension.spec_version);
            }
        }

        'extension: for required_extension in DEVICE_EXTENSIONS.iter() {
            for extension in extension_properties.iter() {
                let extension_name = vk_to_string(&extension.extension_name)?;
                if (*required_extension) == extension_name {
                    continue 'extension;
                }
            }
            return Ok(false);
        }

        Ok(true)
    }

    fn query_swapchain_support(
        physical_device: vk::PhysicalDevice,
        surface_instance: &khr::surface::Instance,
        surface: &vk::SurfaceKHR
    ) -> Result<SwapchainSupportDetails> {
        unsafe {
            let capabilities = surface_instance
                .get_physical_device_surface_capabilities(physical_device, *surface)
                .context("Failed to query Surface Capabilities!")?;
            let formats = surface_instance
                .get_physical_device_surface_formats(physical_device, *surface)
                .context("Failed to query Surface Formats!")?;
            let present_modes = surface_instance
                .get_physical_device_surface_present_modes(physical_device, *surface)
                .context("Failed to query Surface Present Modes!")?;

            Ok(SwapchainSupportDetails { capabilities, formats, present_modes })
        }
    }

    fn create_logical_device(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        surface_instance: &khr::surface::Instance,
        surface: &vk::SurfaceKHR
    ) -> Result<(ash::Device, QueueFamilyIndices)> {
        let indices = Self::find_queue_family(instance, physical_device, surface_instance, surface)?;

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

        let device_extensions_raw: Vec<CString> = DEVICE_EXTENSIONS
            .iter()
            .map(|ext_name| CString::new(*ext_name).unwrap())
            .collect();
        let device_extensions: Vec<*const i8> = device_extensions_raw
            .iter()
            .map(|ext_name| ext_name.as_ptr())
            .collect();

        let mut create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_info)
            .enabled_features(&device_features)
            .enabled_extension_names(&device_extensions);

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

    fn create_swapchain(
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        surface_instance: &khr::surface::Instance,
        surface: &vk::SurfaceKHR
    ) -> Result<(khr::swapchain::Device, vk::SwapchainKHR, vk::Format, vk::Extent2D, Vec<vk::Image>)> {
        let swapchain_support = Self::query_swapchain_support(physical_device, surface_instance, surface)?;

        let surface_format = Self::choose_swapchain_format(&swapchain_support.formats);
        let present_mode = Self::choose_swapchain_present_mode(&swapchain_support.present_modes);
        let extent = Self::choose_swapchain_extent(&swapchain_support.capabilities);

        let image_count = swapchain_support.capabilities.min_image_count + 1;
        let image_count = if swapchain_support.capabilities.max_image_count > 0 {
            image_count.min(swapchain_support.capabilities.max_image_count)
        } else {
            image_count
        };

        let indices = Self::find_queue_family(instance, physical_device, surface_instance, surface)?;
        let families = [indices.graphics_family.unwrap(), indices.present_family.unwrap()];

        let create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(*surface)
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .pre_transform(swapchain_support.capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true);

        let create_info = match indices.graphics_family == indices.present_family {
            true => create_info.image_sharing_mode(vk::SharingMode::EXCLUSIVE),
            false => create_info
                .image_sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(&families),
        };

        let swapchain_instance = khr::swapchain::Device::new(instance, device);
        let swapchain = unsafe {
            swapchain_instance
                .create_swapchain(&create_info, None)
                .context("Failed to create Swapchain!")?
        };
        let swapchain_images = unsafe {
            swapchain_instance
                .get_swapchain_images(swapchain)
                .context("Failed to get Swapchain Images!")?
        };

        Ok((swapchain_instance, swapchain, surface_format.format, extent, swapchain_images))
    }

    fn choose_swapchain_format(available_formats: &Vec<vk::SurfaceFormatKHR>)
        -> vk::SurfaceFormatKHR {
        available_formats
            .iter()
            .find(|format| {
                format.format == vk::Format::B8G8R8A8_SRGB
                    && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .unwrap_or(&available_formats[0])
            .clone()
    }

    fn choose_swapchain_present_mode(available_present_modes: &Vec<vk::PresentModeKHR>)
        -> vk::PresentModeKHR {
        *available_present_modes
            .iter()
            .find(|&&mode| mode == vk::PresentModeKHR::MAILBOX)
            .unwrap_or(&vk::PresentModeKHR::FIFO)
    }

    fn choose_swapchain_extent(capabilities: &vk::SurfaceCapabilitiesKHR) -> vk::Extent2D {
        if capabilities.current_extent.width != u32::max_value() {
            capabilities.current_extent
        } else {
            use num::clamp;

            vk::Extent2D {
                width: clamp(
                    WIDTH,
                    capabilities.min_image_extent.width,
                    capabilities.max_image_extent.width,
                ),
                height: clamp(
                    HEIGHT,
                    capabilities.min_image_extent.height,
                    capabilities.max_image_extent.height,
                ),
            }
        }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe {
            self.swapchain_instance.destroy_swapchain(self.swapchain, None);
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
