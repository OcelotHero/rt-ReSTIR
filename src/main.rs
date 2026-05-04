extern crate winit;

use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::window::{Window, WindowId};

use ash::{ext, khr, vk};

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};

use anyhow::{Context, Result, bail, ensure};

use inline_spirv::include_spirv;

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

const VALIDATION: ValidationInfo = ValidationInfo {
    enabled: true,
    required_validation_layers: ["VK_LAYER_KHRONOS_validation"],
};
#[cfg(not(target_os = "macos"))]
const DEVICE_EXTENSIONS: [&'static str; 1] = ["VK_KHR_swapchain"];
#[cfg(target_os = "macos")]
const DEVICE_EXTENSIONS: [&'static str; 2] = ["VK_KHR_swapchain", "VK_KHR_portability_subset"];

const MAX_FRAMES_IN_FLIGHT: usize = 2;

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

fn populate_debug_messenger_create_info(create_info: &mut vk::DebugUtilsMessengerCreateInfoEXT) {
    create_info.message_severity = vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
        // | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
        // | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
        | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR;
    create_info.message_type = vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
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

    physical_device: vk::PhysicalDevice,
    device: ash::Device,

    _graphics_queue: vk::Queue,
    _present_queue: vk::Queue,

    swapchain_instance: khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
    swapchain_format: vk::Format,
    swapchain_extent: vk::Extent2D,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,
    swapchain_framebuffers: Vec<vk::Framebuffer>,

    render_pass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
    graphics_pipeline: vk::Pipeline,

    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,

    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    current_frame: usize,

    pub framebuffer_resized: bool,
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
            Self::create_logical_device(&instance, &physical_device, &surface_instance, &surface)?;
        let graphics_queue = unsafe {
            device.get_device_queue(family_indices.graphics_family.context("Graphics")?, 0)
        };
        let present_queue = unsafe {
            device.get_device_queue(family_indices.present_family.context("Present")?, 0)
        };

        let (swapchain_instance, swapchain, swapchain_format, extent, swapchain_images) = Self::create_swapchain(
            &instance,
            &device,
            &physical_device,
            &surface_instance,
            &surface,
        )?;
        let swapchain_image_views = Self::create_image_views(&device, swapchain_format, &swapchain_images)?;

        let render_pass = Self::create_render_pass(&device, swapchain_format)?;
        let (pipeline_layout, graphics_pipeline) =
            Self::create_graphics_pipeline(&device, extent, render_pass)?;

        let swapchain_framebuffers =
            Self::create_framebuffers(&device, extent, render_pass, &swapchain_image_views)?;

        let command_pool = Self::create_command_pool(
            &device,
            family_indices.graphics_family.context("Graphics")?,
        )?;
        let command_buffers = Self::create_command_buffers(&device, command_pool)?;

        let (image_available_semaphores, render_finished_semaphores, in_flight_fences) =
            Self::create_sync_objects(&device)?;

        Ok(Engine {
            _entry: entry,
            instance,
            debug_utils_instance,
            debug_messenger,
            surface_instance,
            surface,

            physical_device,
            device,

            _graphics_queue: graphics_queue,
            _present_queue: present_queue,

            swapchain_instance,
            swapchain,
            swapchain_format,
            swapchain_extent: extent,
            swapchain_images,
            swapchain_image_views,
            swapchain_framebuffers,

            render_pass,
            pipeline_layout,
            graphics_pipeline,

            command_pool,
            command_buffers,

            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
            current_frame: 0,

            framebuffer_resized: false,
        })
    }

    fn create_instance(
        entry: &ash::Entry,
        required_extensions: &[*const i8],
    ) -> Result<ash::Instance> {
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
            required_extensions.push(vk::KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_NAME.as_ptr());
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

        #[cfg(target_os = "macos")]
        {
            create_info = create_info.flags(vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR);
        }

        let mut debug_create_info = vk::DebugUtilsMessengerCreateInfoEXT::default();
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

    fn create_surface(
        entry: &ash::Entry,
        instance: &ash::Instance,
        window: &Window,
    ) -> Result<(khr::surface::Instance, vk::SurfaceKHR)> {
        let surface = unsafe {
            ash_window::create_surface(
                entry,
                instance,
                window.display_handle()?.as_raw(),
                window.window_handle()?.as_raw(),
                None,
            )
            .context("Failed to create Surface!")?
        };
        let surface_instance = khr::surface::Instance::new(entry, instance);

        Ok((surface_instance, surface))
    }

    fn setup_debug_messenger(
        entry: &ash::Entry,
        instance: &ash::Instance,
    ) -> Result<(
        Option<ext::debug_utils::Instance>,
        Option<vk::DebugUtilsMessengerEXT>,
    )> {
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
        surface: &vk::SurfaceKHR,
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
                if Self::is_device_suitable(instance, device, surface_instance, surface)? {
                    break 'selection device;
                }
            }
            bail!("Failed to find a suitable GPU!")
        };

        Ok(*physical_device)
    }

    fn is_device_suitable(
        instance: &ash::Instance,
        device: &vk::PhysicalDevice,
        surface_instance: &khr::surface::Instance,
        surface: &vk::SurfaceKHR,
    ) -> Result<bool> {
        let device_properties = unsafe { instance.get_physical_device_properties(*device) };
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
        device: &vk::PhysicalDevice,
        surface_instance: &khr::surface::Instance,
        surface: &vk::SurfaceKHR,
    ) -> Result<QueueFamilyIndices> {
        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(*device) };

        let mut queue_family_indices = QueueFamilyIndices::default();

        for (index, queue_family) in queue_families.iter().enumerate() {
            if queue_family.queue_count > 0
                && queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
            {
                queue_family_indices.graphics_family = Some(index as u32);
            }

            let present_support = unsafe {
                surface_instance.get_physical_device_surface_support(*device, index as u32, *surface)
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

    fn check_device_extension_support(
        instance: &ash::Instance,
        device: &vk::PhysicalDevice,
    ) -> Result<bool> {
        let extension_properties = unsafe {
            instance
                .enumerate_device_extension_properties(*device)
                .context("Failed to enumerate Device Extension Properties!")?
        };

        if extension_properties.is_empty() {
            eprintln!("No available device extensions.");
            return Ok(false);
        } else {
            println!("Available Device Extensions: ");
            for extension in extension_properties.iter() {
                let extension_name = vk_to_string(&extension.extension_name)?;
                println!(
                    "\t\tName: {}, Version: {}",
                    extension_name, extension.spec_version
                );
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
        physical_device: &vk::PhysicalDevice,
        surface_instance: &khr::surface::Instance,
        surface: &vk::SurfaceKHR,
    ) -> Result<SwapchainSupportDetails> {
        unsafe {
            let capabilities = surface_instance
                .get_physical_device_surface_capabilities(*physical_device, *surface)
                .context("Failed to query Surface Capabilities!")?;
            let formats = surface_instance
                .get_physical_device_surface_formats(*physical_device, *surface)
                .context("Failed to query Surface Formats!")?;
            let present_modes = surface_instance
                .get_physical_device_surface_present_modes(*physical_device, *surface)
                .context("Failed to query Surface Present Modes!")?;

            Ok(SwapchainSupportDetails {
                capabilities,
                formats,
                present_modes,
            })
        }
    }

    fn create_logical_device(
        instance: &ash::Instance,
        physical_device: &vk::PhysicalDevice,
        surface_instance: &khr::surface::Instance,
        surface: &vk::SurfaceKHR,
    ) -> Result<(ash::Device, QueueFamilyIndices)> {
        let indices =
            Self::find_queue_family(instance, physical_device, surface_instance, surface)?;

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
                .create_device(*physical_device, &create_info, None)
                .context("Failed to create Logical Device!")?
        };

        Ok((device, indices))
    }

    fn create_swapchain(
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: &vk::PhysicalDevice,
        surface_instance: &khr::surface::Instance,
        surface: &vk::SurfaceKHR,
    ) -> Result<(
        khr::swapchain::Device,
        vk::SwapchainKHR,
        vk::Format,
        vk::Extent2D,
        Vec<vk::Image>,
    )> {
        let swapchain_support =
            Self::query_swapchain_support(physical_device, surface_instance, surface)?;

        let surface_format = Self::choose_swapchain_format(&swapchain_support.formats);
        let present_mode = Self::choose_swapchain_present_mode(&swapchain_support.present_modes);
        let extent = Self::choose_swapchain_extent(&swapchain_support.capabilities);

        let image_count = swapchain_support.capabilities.min_image_count + 1;
        let image_count = if swapchain_support.capabilities.max_image_count > 0 {
            image_count.min(swapchain_support.capabilities.max_image_count)
        } else {
            image_count
        };

        let indices =
            Self::find_queue_family(instance, physical_device, surface_instance, surface)?;
        let families = [
            indices.graphics_family.unwrap(),
            indices.present_family.unwrap(),
        ];

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

        Ok((
            swapchain_instance,
            swapchain,
            surface_format.format,
            extent,
            swapchain_images,
        ))
    }

    fn choose_swapchain_format(
        available_formats: &Vec<vk::SurfaceFormatKHR>,
    ) -> vk::SurfaceFormatKHR {
        available_formats
            .iter()
            .find(|format| {
                format.format == vk::Format::B8G8R8A8_SRGB
                    && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .unwrap_or(&available_formats[0])
            .clone()
    }

    fn choose_swapchain_present_mode(
        available_present_modes: &Vec<vk::PresentModeKHR>,
    ) -> vk::PresentModeKHR {
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

    fn create_image_views(
        device: &ash::Device,
        format: vk::Format,
        images: &Vec<vk::Image>,
    ) -> Result<Vec<vk::ImageView>> {
        let mut image_views = Vec::with_capacity(images.len());

        for &image in images.iter() {
            let create_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format)
                .components(vk::ComponentMapping::default())
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            let image_view = unsafe {
                device
                    .create_image_view(&create_info, None)
                    .context("Failed to create Image View!")?
            };
            image_views.push(image_view);
        }

        Ok(image_views)
    }

    fn create_render_pass(device: &ash::Device, format: vk::Format) -> Result<vk::RenderPass> {
        let color_attachment = [vk::AttachmentDescription::default()
            .format(format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR)];

        let color_attachment_ref = [vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        }];

        let subpass = [vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_attachment_ref)];

        let dependency = [vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)];

        let render_pass_info = vk::RenderPassCreateInfo::default()
            .attachments(&color_attachment)
            .subpasses(&subpass)
            .dependencies(&dependency);

        let render_pass = unsafe {
            device
                .create_render_pass(&render_pass_info, None)
                .context("Failed to create Render Pass!")?
        };

        Ok(render_pass)
    }

    fn create_graphics_pipeline(
        device: &ash::Device,
        swapchain_extent: vk::Extent2D,
        render_pass: vk::RenderPass,
    ) -> Result<(vk::PipelineLayout, vk::Pipeline)> {
        let vert_shader_code = include_spirv!("shaders/shader.vert", vert);
        let frag_shader_code = include_spirv!("shaders/shader.frag", frag);

        let vert_shader_module = Self::create_shader_module(device, vert_shader_code)?;
        let frag_shader_module = Self::create_shader_module(device, frag_shader_code)?;

        let main_function_name = CStr::from_bytes_with_nul(b"main\0")?;

        let shader_stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_shader_module)
                .name(main_function_name),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag_shader_module)
                .name(main_function_name),
        ];

        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default();
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let viewport = [vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: WIDTH as f32,
            height: HEIGHT as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }];
        let scissor = [vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: swapchain_extent,
        }];
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            // .viewport_count(viewport.len() as u32)
            // .scissor_count(scissor.len() as u32);
            .viewports(&viewport)
            .scissors(&scissor);

        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::CLOCKWISE)
            .depth_bias_enable(false);

        let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let color_blend_attachment = [vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(
                vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A,
            )
            .blend_enable(false)];
        let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .attachments(&color_blend_attachment);

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default();
        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .context("Failed to create Pipeline Layout!")?
        };

        let pipeline_info = [vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blending)
            .dynamic_state(&dynamic_state)
            .layout(pipeline_layout)
            .render_pass(render_pass)
            .subpass(0)];

        let graphics_pipelines = unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_info, None)
                .map_err(|(_, err)| err)
                .context("Failed to create Graphics Pipeline!")?
        };

        unsafe {
            device.destroy_shader_module(vert_shader_module, None);
            device.destroy_shader_module(frag_shader_module, None);
        }

        Ok((pipeline_layout, graphics_pipelines[0]))
    }

    fn create_shader_module(device: &ash::Device, code: &[u32]) -> Result<vk::ShaderModule> {
        let create_info = vk::ShaderModuleCreateInfo::default().code(code);

        let shader_module = unsafe {
            device
                .create_shader_module(&create_info, None)
                .context("Failed to create Shader Module!")?
        };

        Ok(shader_module)
    }

    fn create_framebuffers(
        device: &ash::Device,
        extent: vk::Extent2D,
        render_pass: vk::RenderPass,
        image_views: &Vec<vk::ImageView>,
    ) -> Result<Vec<vk::Framebuffer>> {
        let mut framebuffers = Vec::with_capacity(image_views.len());

        for &image_view in image_views.iter() {
            let attachments = [image_view];
            let framebuffer_info = vk::FramebufferCreateInfo::default()
                .render_pass(render_pass)
                .attachments(&attachments)
                .width(extent.width)
                .height(extent.height)
                .layers(1);

            let framebuffer = unsafe {
                device
                    .create_framebuffer(&framebuffer_info, None)
                    .context("Failed to create Framebuffer!")?
            };
            framebuffers.push(framebuffer);
        }

        Ok(framebuffers)
    }

    fn create_command_pool(
        device: &ash::Device,
        graphics_family_index: u32,
    ) -> Result<vk::CommandPool> {
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(graphics_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        let command_pool = unsafe {
            device
                .create_command_pool(&pool_info, None)
                .context("Failed to create Command Pool!")?
        };

        Ok(command_pool)
    }

    fn create_command_buffers(
        device: &ash::Device,
        command_pool: vk::CommandPool,
    ) -> Result<Vec<vk::CommandBuffer>> {
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(MAX_FRAMES_IN_FLIGHT as u32);

        let command_buffers = unsafe {
            device
                .allocate_command_buffers(&allocate_info)
                .context("Failed to allocate Command Buffers!")?
        };

        Ok(command_buffers)
    }

    fn record_command_buffer(
        &self,
        command_buffer: vk::CommandBuffer,
        image_index: u32,
    ) -> Result<()> {
        let begin_info = vk::CommandBufferBeginInfo::default();

        unsafe {
            self.device
                .begin_command_buffer(command_buffer, &begin_info)
                .context("Failed to begin recording Command Buffer!")?;

            let clear_values = [vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            }];
            let render_pass_info = vk::RenderPassBeginInfo::default()
                .render_pass(self.render_pass)
                .framebuffer(self.swapchain_framebuffers[image_index as usize])
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: self.swapchain_extent,
                })
                .clear_values(&clear_values);

            self.device.cmd_begin_render_pass(
                command_buffer,
                &render_pass_info,
                vk::SubpassContents::INLINE,
            );
            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.graphics_pipeline,
            );
            self.device.cmd_set_viewport(
                command_buffer,
                0,
                &[vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: self.swapchain_extent.width as f32,
                    height: self.swapchain_extent.height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            );
            self.device.cmd_set_scissor(
                command_buffer,
                0,
                &[vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: self.swapchain_extent,
                }],
            );
            self.device.cmd_draw(command_buffer, 3, 1, 0, 0);
            self.device.cmd_end_render_pass(command_buffer);

            self.device
                .end_command_buffer(command_buffer)
                .context("Failed to record Command Buffer!")?;
        }

        Ok(())
    }

    fn create_sync_objects(
        device: &ash::Device,
    ) -> Result<(Vec<vk::Semaphore>, Vec<vk::Semaphore>, Vec<vk::Fence>)> {
        let semaphore_info = vk::SemaphoreCreateInfo::default();
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

        let mut image_available_semaphores = Vec::new();
        let mut render_finished_semaphores = Vec::new();
        let mut in_flight_fences = Vec::new();

        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            unsafe {
                let image_available_semaphore = device
                    .create_semaphore(&semaphore_info, None)
                    .context("Failed to create Image Available Semaphore!")?;
                let render_finished_semaphore = device
                    .create_semaphore(&semaphore_info, None)
                    .context("Failed to create Render Finished Semaphore!")?;
                let in_flight_fence = device
                    .create_fence(&fence_info, None)
                    .context("Failed to create In Flight Fence!")?;
                image_available_semaphores.push(image_available_semaphore);
                render_finished_semaphores.push(render_finished_semaphore);
                in_flight_fences.push(in_flight_fence);
            };
        }

        Ok((
            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
        ))
    }

    pub fn draw_frame(&mut self) -> Result<()> {
        unsafe {
            let wait_fences = [self.in_flight_fences[self.current_frame]];
            self.device
                .wait_for_fences(&wait_fences, true, u64::MAX)
                .context("Failed to wait for In Flight Fence!")?;

            let (image_index, is_suboptimal) = match self.swapchain_instance.acquire_next_image(
                self.swapchain,
                u64::MAX,
                self.image_available_semaphores[self.current_frame],
                vk::Fence::null(),
            ) {
                Ok(result) => result,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => return self.recreate_swapchain(),
                Err(err) => return Err(err).context("Failed to acquire next image from Swapchain!"),
            };

            self.device
                .reset_fences(&wait_fences)
                .context("Failed to reset In Flight Fence!")?;

            self.device
                .reset_command_buffer(
                    self.command_buffers[self.current_frame],
                    vk::CommandBufferResetFlags::empty(),
                )
                .context("Failed to reset Command Buffer!")?;
            self.record_command_buffer(self.command_buffers[self.current_frame], image_index)?;

            let wait_semaphores = [self.image_available_semaphores[self.current_frame]];
            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];

            let command_buffers = [self.command_buffers[self.current_frame]];
            let signal_semaphores = [self.render_finished_semaphores[self.current_frame]];
            let submit_info = [vk::SubmitInfo::default()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(&command_buffers)
                .signal_semaphores(&signal_semaphores)];
            self.device
                .queue_submit(
                    self._graphics_queue,
                    &submit_info,
                    self.in_flight_fences[self.current_frame],
                )
                .context("Failed to submit draw command buffer!")?;

            let swapchains = [self.swapchain];
            let image_indices = [image_index];
            let changed = match self.swapchain_instance
                .queue_present(
                    self._present_queue,
                    &vk::PresentInfoKHR::default()
                        .wait_semaphores(&signal_semaphores)
                        .swapchains(&swapchains)
                        .image_indices(&image_indices),
                ) {
                Ok(suboptimal) => suboptimal || is_suboptimal,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::SUBOPTIMAL_KHR) => true,
                Err(err) => {
                    return Err(err).context("Failed to present Swapchain image!");
                }
            };

            if changed || self.framebuffer_resized {
                self.framebuffer_resized = false;
                self.recreate_swapchain()?;
            }

            self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
        }

        Ok(())
    }

    fn recreate_swapchain(&mut self) -> Result<()> {
        unsafe {
            self.device
                .device_wait_idle()
                .context("Failed to wait device idle!")?;

            self.cleanup_swapchain();

            let (
                swapchain_instance,
                swapchain,
                swapchain_format,
                swapchain_extent,
                swapchain_images,
            ) = Self::create_swapchain(
                &self.instance,
                &self.device,
                &self.physical_device,
                &self.surface_instance,
                &self.surface,
            )?;
            self.swapchain_instance = swapchain_instance;
            self.swapchain = swapchain;
            self.swapchain_format = swapchain_format;
            self.swapchain_extent = swapchain_extent;
            self.swapchain_images = swapchain_images;

            self.swapchain_image_views = Self::create_image_views(
                &self.device,
                self.swapchain_format,
                &self.swapchain_images,
            )?;

            self.render_pass = Self::create_render_pass(&self.device, self.swapchain_format)?;

            let (pipeline_layout, graphics_pipeline) = Self::create_graphics_pipeline(
                &self.device,
                self.swapchain_extent,
                self.render_pass,
            )?;
            self.pipeline_layout = pipeline_layout;
            self.graphics_pipeline = graphics_pipeline;

            self.swapchain_framebuffers = Self::create_framebuffers(
                &self.device,
                self.swapchain_extent,
                self.render_pass,
                &self.swapchain_image_views,
            )?;

            self.command_buffers = Self::create_command_buffers(&self.device, self.command_pool)?;
        }

        Ok(())
    }

    fn cleanup_swapchain(&mut self) {
        unsafe {
            self.device.free_command_buffers(self.command_pool, &self.command_buffers);

            for &framebuffer in self.swapchain_framebuffers.iter() {
                self.device.destroy_framebuffer(framebuffer, None);
            }

            self.device.destroy_pipeline(self.graphics_pipeline, None);
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);

            self.device.destroy_render_pass(self.render_pass, None);

            for &image_view in self.swapchain_image_views.iter() {
                self.device.destroy_image_view(image_view, None);
            }
            self.swapchain_instance
                .destroy_swapchain(self.swapchain, None);
        }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe {
            self.device
                .device_wait_idle()
                .expect("Failed to wait device idle!");

            self.cleanup_swapchain();

            for i in 0..MAX_FRAMES_IN_FLIGHT {
                self.device
                    .destroy_semaphore(self.image_available_semaphores[i], None);
                self.device
                    .destroy_semaphore(self.render_finished_semaphores[i], None);
                self.device.destroy_fence(self.in_flight_fences[i], None);
            }

            self.device.destroy_command_pool(self.command_pool, None);

            self.device.destroy_device(None);
            if VALIDATION.enabled {
                if let (Some(loader), Some(messenger)) =
                    (&self.debug_utils_instance, &self.debug_messenger)
                {
                    loader.destroy_debug_utils_messenger(*messenger, None);
                }
            }
            self.surface_instance.destroy_surface(self.surface, None);
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
        if self.window.is_some() { return; }

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

        self.engine = Some(
            Engine::new(&window, &required_extensions)
                .expect("Failed to initialize Vulkan Engine!"),
        );
        self.window = Some(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                println!("The close button was pressed; stopping");
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                self.window.as_ref().unwrap().request_redraw();
                self.engine
                    .as_mut()
                    .unwrap()
                    .draw_frame()
                    .expect("Failed to draw frame!");
            }
            WindowEvent::Resized(_size) => {
                self.engine.as_mut().unwrap().framebuffer_resized = true;
            }
            _ => (),
        }
    }
}

fn main() {
    // If running under WSL, force winit to use X11 (Wayland support on WSL is unreliable).
    #[cfg(target_os = "linux")]
    {
        fn is_wsl() -> bool {
            if std::env::var_os("WSL_DISTRO_NAME").is_some() {
                return true;
            }
            if let Ok(ver) = std::fs::read_to_string("/proc/version") {
                let v = ver.to_lowercase();
                if v.contains("microsoft") || v.contains("wsl") {
                    return true;
                }
            }
            if let Ok(release) = std::fs::read_to_string("/proc/sys/kernel/osrelease") {
                let r = release.to_lowercase();
                if r.contains("microsoft") || r.contains("wsl") {
                    return true;
                }
            }
            false
        }

        if is_wsl() {
            unsafe { std::env::set_var("WINIT_UNIX_BACKEND", "x11") };
            unsafe { std::env::remove_var("WAYLAND_DISPLAY") };
            println!("[diagnostic] Detected WSL: forcing WINIT_UNIX_BACKEND=x11 and unsetting WAYLAND_DISPLAY");
        }
    }

    let event_loop = EventLoop::new().unwrap();

    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.set_control_flow(ControlFlow::Wait);

    let mut app = HelloTriangleApplication::default();
    let _ = event_loop.run_app(&mut app);
}
