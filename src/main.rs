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
use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};

use inline_spirv::include_spirv;

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

const MAX_FRAMES_IN_FLIGHT: usize = 2;
struct ValidationInfo {
    enabled: bool,
    required_validation_layers: [&'static str; 1],
}
const VALIDATION: ValidationInfo =
    ValidationInfo { enabled: true, required_validation_layers: ["VK_LAYER_KHRONOS_validation"] };

#[cfg(not(target_os = "macos"))]
const DEVICE_EXTENSIONS: [&'static str; 1] = ["VK_KHR_swapchain"];
#[cfg(target_os = "macos")]
const DEVICE_EXTENSIONS: [&'static str; 2] = ["VK_KHR_swapchain", "VK_KHR_portability_subset"];

#[derive(Default, Clone, Copy)]
struct QueueFamilyIndices {
    graphics_family: Option<u32>,
    present_family: Option<u32>,
}

impl QueueFamilyIndices {
    pub fn find_queue_family(
        instance: &ash::Instance,
        device: &vk::PhysicalDevice,
        surface_loader: &khr::surface::Instance,
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
                surface_loader.get_physical_device_surface_support(*device, index as u32, *surface)
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

    pub fn is_complete(&self) -> bool {
        self.graphics_family.is_some() && self.present_family.is_some()
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

fn vk_to_string(raw_string_array: &[c_char]) -> Result<String> {
    let raw_string = unsafe {
        let pointer = raw_string_array.as_ptr();
        CStr::from_ptr(pointer)
    };

    Ok(raw_string.to_str().context("Failed to convert vulkan raw string.")?.to_owned())
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

struct VulkanContext {
    entry: ash::Entry,
    instance: ash::Instance,
    surface_loader: khr::surface::Instance,
    surface: vk::SurfaceKHR,
    debug_utils: Option<(ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
}

impl VulkanContext {
    pub fn init(window: &Window, required_extensions: &[*const i8]) -> Result<Self> {
        let entry = ash::Entry::linked();
        let instance = Self::create_instance(&entry, required_extensions)?;
        let surface_loader = khr::surface::Instance::new(&entry, &instance);

        let mut context = Self {
            entry,
            instance: instance,
            surface_loader,
            surface: vk::SurfaceKHR::null(),
            debug_utils: None,
        };
        context.surface = Self::create_surface(&context.entry, &context.instance, window)?;
        context.debug_utils = Self::setup_debug_messenger(&context.entry, &context.instance)?;

        Ok(context)
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
            .application_version(vk::make_api_version(
                0,
                env!("CARGO_PKG_VERSION_MAJOR").parse()?,
                env!("CARGO_PKG_VERSION_MINOR").parse()?,
                env!("CARGO_PKG_VERSION_PATCH").parse()?,
            ))
            .engine_name(&engine_name)
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::API_VERSION_1_0);

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
        let validation_layers: Vec<*const i8> =
            validation_layers_raw.iter().map(|layer_name| layer_name.as_ptr()).collect();

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

    fn setup_debug_messenger(
        entry: &ash::Entry,
        instance: &ash::Instance,
    ) -> Result<Option<(ext::debug_utils::Instance, vk::DebugUtilsMessengerEXT)>> {
        if !VALIDATION.enabled {
            return Ok(None);
        }

        let mut create_info = vk::DebugUtilsMessengerCreateInfoEXT::default();
        populate_debug_messenger_create_info(&mut create_info);

        let loader = ext::debug_utils::Instance::new(entry, instance);
        let messenger = unsafe {
            loader
                .create_debug_utils_messenger(&create_info, None)
                .context("Failed to create Debug Utils Messenger!")?
        };

        Ok(Some((loader, messenger)))
    }

    fn create_surface(
        entry: &ash::Entry,
        instance: &ash::Instance,
        window: &Window,
    ) -> Result<vk::SurfaceKHR> {
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

        Ok(surface)
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            if let Some((loader, messenger)) = self.debug_utils.take() {
                loader.destroy_debug_utils_messenger(messenger, None);
            }
            if self.surface != vk::SurfaceKHR::null() {
                self.surface_loader.destroy_surface(self.surface, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

struct DeviceContext {
    vk_ctx: Arc<VulkanContext>,
    handle: ash::Device,
    device: vk::PhysicalDevice,
    indices: QueueFamilyIndices,
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
}

impl DeviceContext {
    pub fn init(vk_ctx: Arc<VulkanContext>) -> Result<Self> {
        let instance = &vk_ctx.instance;
        let surface_loader = &vk_ctx.surface_loader;
        let surface = &vk_ctx.surface;

        let (indices, device) = Self::pick_physical_device(instance, surface_loader, surface)?;
        let mut device = Self {
            vk_ctx: vk_ctx.clone(),
            device,
            indices,
            handle: Self::create_logical_device(instance, &device, &indices)?,
            graphics_queue: vk::Queue::null(),
            present_queue: vk::Queue::null(),
        };

        device.graphics_queue =
            unsafe { device.handle.get_device_queue(indices.graphics_family.unwrap(), 0) };
        device.present_queue =
            unsafe { device.handle.get_device_queue(indices.present_family.unwrap(), 0) };

        Ok(device)
    }

    fn pick_physical_device(
        instance: &ash::Instance,
        surface_loader: &khr::surface::Instance,
        surface: &vk::SurfaceKHR,
    ) -> Result<(QueueFamilyIndices, vk::PhysicalDevice)> {
        let physical_devices = unsafe {
            instance
                .enumerate_physical_devices()
                .context("Failed to enumerate Physical Devices!")?
        };

        println!("{} devices (GPU) found with vulkan support.", physical_devices.len());

        let (indices, device) = 'selection: {
            for device in physical_devices.iter() {
                if let (indices, true) =
                    Self::is_device_suitable(instance, device, surface_loader, surface)?
                {
                    break 'selection (indices, *device);
                }
            }
            bail!("Failed to find a suitable GPU!")
        };

        Ok((indices, device))
    }

    fn is_device_suitable(
        instance: &ash::Instance,
        device: &vk::PhysicalDevice,
        surface_loader: &khr::surface::Instance,
        surface: &vk::SurfaceKHR,
    ) -> Result<(QueueFamilyIndices, bool)> {
        let device_properties = unsafe { instance.get_physical_device_properties(*device) };
        let device_name = vk_to_string(&device_properties.device_name)?;
        println!("\tDevice Name: {}", device_name);

        let indices =
            QueueFamilyIndices::find_queue_family(instance, device, surface_loader, surface)?;
        let extensions_supported = Self::check_device_extension_support(instance, device)?;
        let swapchain_adequate = extensions_supported && {
            let (_, formats, present_modes) =
                SwapchainBundle::query_swapchain_support(device, surface_loader, surface)?;
            !formats.is_empty() && !present_modes.is_empty()
        };

        Ok((indices, indices.is_complete() && extensions_supported && swapchain_adequate))
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

    fn create_logical_device(
        instance: &ash::Instance,
        physical_device: &vk::PhysicalDevice,
        indices: &QueueFamilyIndices,
    ) -> Result<ash::Device> {
        let unique_queue_families = [indices.graphics_family, indices.present_family]
            .iter()
            .filter_map(|&family| family)
            .collect::<std::collections::HashSet<u32>>();

        let queue_priority = [1.0_f32];
        let queue_create_infos = unique_queue_families
            .iter()
            .map(|&family| {
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(family)
                    .queue_priorities(&queue_priority)
            })
            .collect::<Vec<_>>();

        let device_features = vk::PhysicalDeviceFeatures::default();

        let device_extensions_raw: Vec<CString> =
            DEVICE_EXTENSIONS.iter().map(|ext_name| CString::new(*ext_name).unwrap()).collect();
        let device_extensions: Vec<*const i8> =
            device_extensions_raw.iter().map(|ext_name| ext_name.as_ptr()).collect();

        let mut create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_create_infos)
            .enabled_features(&device_features)
            .enabled_extension_names(&device_extensions);

        let validation_layers_raw: Vec<CString> = VALIDATION
            .required_validation_layers
            .iter()
            .map(|layer_name| CString::new(*layer_name).unwrap())
            .collect();
        let validation_layers: Vec<*const i8> =
            validation_layers_raw.iter().map(|layer_name| layer_name.as_ptr()).collect();

        if VALIDATION.enabled {
            create_info = create_info.enabled_layer_names(&validation_layers);
        }

        let device = unsafe {
            instance
                .create_device(*physical_device, &create_info, None)
                .context("Failed to create Logical Device!")?
        };

        Ok(device)
    }
}

impl Drop for DeviceContext {
    fn drop(&mut self) {
        unsafe {
            self.handle.device_wait_idle().ok();
            self.handle.destroy_device(None);
        }
    }
}

struct Defer<F: FnOnce()>(Option<F>);

impl<F: FnOnce()> Defer<F> {
    pub fn new(f: F) -> Self {
        Self(Some(f))
    }
}

impl<F: FnOnce()> Drop for Defer<F> {
    fn drop(&mut self) {
        if let Some(f) = self.0.take() {
            f();
        }
    }
}

struct PipelineBundle {
    dev_ctx: Arc<DeviceContext>,
    render_pass: vk::RenderPass,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

impl PipelineBundle {
    pub fn new(dev_ctx: Arc<DeviceContext>) -> Result<Self> {
        let handle = &dev_ctx.handle;
        let device = &dev_ctx.device;
        let surface_loader = &dev_ctx.vk_ctx.surface_loader;
        let surface = &dev_ctx.vk_ctx.surface;

        let (_, formats, _) =
            SwapchainBundle::query_swapchain_support(device, surface_loader, surface)?;
        let format = SwapchainBundle::choose_swapchain_format(&formats).format;

        let mut bundle = Self {
            dev_ctx: dev_ctx.clone(),
            render_pass: Self::create_render_pass(handle, format)?,
            layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
        };
        bundle.layout = Self::create_pipeline_layout(handle)?;
        bundle.pipeline =
            Self::create_graphics_pipeline(handle, &bundle.render_pass, &bundle.layout)?;

        Ok(bundle)
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

    fn create_pipeline_layout(device: &ash::Device) -> Result<vk::PipelineLayout> {
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default();

        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .context("Failed to create Pipeline Layout!")?
        };

        Ok(pipeline_layout)
    }

    fn create_graphics_pipeline(
        device: &ash::Device,
        render_pass: &vk::RenderPass,
        pipeline_layout: &vk::PipelineLayout,
    ) -> Result<vk::Pipeline> {
        let vert_code = include_spirv!("shaders/shader.vert", vert);
        let frag_code = include_spirv!("shaders/shader.frag", frag);

        let vert_module = Self::create_shader_module(device, vert_code)?;
        let _vert_defer =
            Defer::new(|| unsafe { device.destroy_shader_module(vert_module, None) });
        let frag_module = Self::create_shader_module(device, frag_code)?;
        let _frag_defer =
            Defer::new(|| unsafe { device.destroy_shader_module(frag_module, None) });

        let main_function_name = CStr::from_bytes_with_nul(b"main\0")?;

        let shader_stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_module)
                .name(main_function_name),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag_module)
                .name(main_function_name),
        ];

        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default();
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let viewport_state =
            vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);

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

        let pipeline_info = [vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .color_blend_state(&color_blending)
            .dynamic_state(&dynamic_state)
            .layout(*pipeline_layout)
            .render_pass(*render_pass)
            .subpass(0)];

        let graphics_pipelines = unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_info, None)
                .map_err(|(_, err)| err)
                .context("Failed to create Graphics Pipeline!")?
        };

        Ok(graphics_pipelines[0])
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
}

impl Drop for PipelineBundle {
    fn drop(&mut self) {
        unsafe {
            self.dev_ctx.handle.destroy_pipeline(self.pipeline, None);
            if self.layout != vk::PipelineLayout::null() {
                self.dev_ctx.handle.destroy_pipeline_layout(self.layout, None);
            }
            if self.render_pass != vk::RenderPass::null() {
                self.dev_ctx.handle.destroy_render_pass(self.render_pass, None);
            }
        }
    }
}

struct SwapchainBundle {
    dev_ctx: Arc<DeviceContext>,
    loader: khr::swapchain::Device,
    handle: vk::SwapchainKHR,
    extent: vk::Extent2D,
    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
    framebuffers: Vec<vk::Framebuffer>,
}

impl SwapchainBundle {
    fn new(dev_ctx: Arc<DeviceContext>, render_pass: &vk::RenderPass) -> Result<Self> {
        let physical_device = &dev_ctx.device;
        let device = &dev_ctx.handle;
        let indices = &dev_ctx.indices;
        let instance = &dev_ctx.vk_ctx.instance;
        let surface_loader = &dev_ctx.vk_ctx.surface_loader;
        let surface = &dev_ctx.vk_ctx.surface;

        let loader = khr::swapchain::Device::new(instance, device);

        let (handle, format, extent, images) =
            Self::create_swapchain(physical_device, surface_loader, surface, indices, &loader)?;

        let mut bundle = Self {
            dev_ctx: dev_ctx.clone(),
            loader,
            handle,
            extent,
            images,
            image_views: Vec::new(),
            framebuffers: Vec::new(),
        };
        bundle.image_views = Self::create_image_views(&device, &format, &bundle.images)?;
        bundle.framebuffers = Self::create_framebuffers(
            &bundle.dev_ctx.handle,
            &bundle.extent,
            render_pass,
            &bundle.image_views,
        )?;

        Ok(bundle)
    }

    pub fn query_swapchain_support(
        physical_device: &vk::PhysicalDevice,
        surface_loader: &khr::surface::Instance,
        surface: &vk::SurfaceKHR,
    ) -> Result<(vk::SurfaceCapabilitiesKHR, Vec<vk::SurfaceFormatKHR>, Vec<vk::PresentModeKHR>)>
    {
        unsafe {
            let capabilities = surface_loader
                .get_physical_device_surface_capabilities(*physical_device, *surface)
                .context("Failed to query Surface Capabilities!")?;
            let formats = surface_loader
                .get_physical_device_surface_formats(*physical_device, *surface)
                .context("Failed to query Surface Formats!")?;
            let present_modes = surface_loader
                .get_physical_device_surface_present_modes(*physical_device, *surface)
                .context("Failed to query Surface Present Modes!")?;

            Ok((capabilities, formats, present_modes))
        }
    }

    fn create_swapchain(
        physical_device: &vk::PhysicalDevice,
        surface_loader: &khr::surface::Instance,
        surface: &vk::SurfaceKHR,
        indices: &QueueFamilyIndices,
        loader: &khr::swapchain::Device,
    ) -> Result<(vk::SwapchainKHR, vk::Format, vk::Extent2D, Vec<vk::Image>)> {
        let (capabilities, formats, present_modes) =
            Self::query_swapchain_support(physical_device, surface_loader, surface)?;

        let format = Self::choose_swapchain_format(&formats);
        let mode = Self::choose_swapchain_present_mode(&present_modes);
        let extent = Self::choose_swapchain_extent(&capabilities);

        let image_count = capabilities.min_image_count + 1;
        let image_count = if capabilities.max_image_count > 0 {
            image_count.min(capabilities.max_image_count)
        } else {
            image_count
        };

        let families = [indices.graphics_family.unwrap(), indices.present_family.unwrap()];

        let create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(*surface)
            .min_image_count(image_count)
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .pre_transform(capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(mode)
            .clipped(true);

        let create_info = match indices.graphics_family == indices.present_family {
            true => create_info.image_sharing_mode(vk::SharingMode::EXCLUSIVE),
            false => create_info
                .image_sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(&families),
        };

        let swapchain = unsafe {
            loader.create_swapchain(&create_info, None).context("Failed to create Swapchain!")?
        };
        let swapchain_images = unsafe {
            loader.get_swapchain_images(swapchain).context("Failed to get Swapchain Images!")?
        };

        Ok((swapchain, format.format, extent, swapchain_images))
    }

    pub fn choose_swapchain_format(
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
        format: &vk::Format,
        images: &Vec<vk::Image>,
    ) -> Result<Vec<vk::ImageView>> {
        let mut image_views = Vec::with_capacity(images.len());

        for &image in images.iter() {
            let create_info = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(*format)
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

    fn create_framebuffers(
        device: &ash::Device,
        extent: &vk::Extent2D,
        render_pass: &vk::RenderPass,
        image_views: &Vec<vk::ImageView>,
    ) -> Result<Vec<vk::Framebuffer>> {
        let mut framebuffers = Vec::with_capacity(image_views.len());

        for &image_view in image_views.iter() {
            let attachments = [image_view];
            let framebuffer_info = vk::FramebufferCreateInfo::default()
                .render_pass(*render_pass)
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
}

impl Drop for SwapchainBundle {
    fn drop(&mut self) {
        unsafe {
            for &framebuffer in self.framebuffers.iter() {
                self.dev_ctx.handle.destroy_framebuffer(framebuffer, None);
            }
            for &image_view in self.image_views.iter() {
                self.dev_ctx.handle.destroy_image_view(image_view, None);
            }
            self.loader.destroy_swapchain(self.handle, None);
        }
    }
}

struct CommandBundle {
    dev_ctx: Arc<DeviceContext>,
    pool: vk::CommandPool,
    buffers: Vec<vk::CommandBuffer>,
}

impl CommandBundle {
    fn new(dev_ctx: Arc<DeviceContext>, max_frames: usize) -> Result<Self> {
        let mut bundle = Self {
            dev_ctx: dev_ctx.clone(),
            pool: Self::create_command_pool(
                &dev_ctx.handle,
                dev_ctx.indices.graphics_family.unwrap(),
            )?,
            buffers: Vec::new(),
        };
        bundle.buffers =
            Self::create_command_buffers(&bundle.dev_ctx.handle, &bundle.pool, max_frames)?;

        Ok(bundle)
    }

    fn create_command_pool(device: &ash::Device, index: u32) -> Result<vk::CommandPool> {
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(index)
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
        command_pool: &vk::CommandPool,
        max_frames: usize,
    ) -> Result<Vec<vk::CommandBuffer>> {
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(*command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(max_frames as u32);

        let command_buffers = unsafe {
            device
                .allocate_command_buffers(&allocate_info)
                .context("Failed to allocate Command Buffers!")?
        };

        Ok(command_buffers)
    }

    pub fn begin_recording(&self, frame_index: usize) -> Result<vk::CommandBuffer> {
        let cmd = self.buffers[frame_index];
        let begin_info = vk::CommandBufferBeginInfo::default();

        unsafe {
            self.dev_ctx
                .handle
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())
                .context("Failed to reset Command Buffer!")?;
            self.dev_ctx
                .handle
                .begin_command_buffer(cmd, &begin_info)
                .context("Failed to begin recording Command Buffer!")?;
        }

        Ok(cmd)
    }

    pub fn end_recording(&self, cmd: vk::CommandBuffer) -> Result<vk::CommandBuffer> {
        unsafe {
            self.dev_ctx.handle.end_command_buffer(cmd)?;
        }

        Ok(cmd)
    }
}

impl Drop for CommandBundle {
    fn drop(&mut self) {
        unsafe {
            self.dev_ctx.handle.destroy_command_pool(self.pool, None);
        }
    }
}

struct SyncBundle {
    dev_ctx: Arc<DeviceContext>,
    s_image_available: Vec<vk::Semaphore>,
    s_render_finished: Vec<vk::Semaphore>,
    f_in_flight: Vec<vk::Fence>,
}

impl SyncBundle {
    fn new(dev_ctx: Arc<DeviceContext>, max_frames: usize) -> Result<Self> {
        let mut bundle = Self {
            dev_ctx: dev_ctx.clone(),
            s_image_available: Vec::with_capacity(max_frames),
            s_render_finished: Vec::with_capacity(max_frames),
            f_in_flight: Vec::with_capacity(max_frames),
        };

        let semaphore_info = vk::SemaphoreCreateInfo::default();
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

        for i in 0..max_frames {
            unsafe {
                bundle.s_image_available.push(
                    dev_ctx
                        .handle
                        .create_semaphore(&semaphore_info, None)
                        .context(format!("Failed to create Image Available Semaphore {}!", i))?,
                );
                bundle.s_render_finished.push(
                    dev_ctx
                        .handle
                        .create_semaphore(&semaphore_info, None)
                        .context(format!("Failed to create Render Finished Semaphore {}!", i))?,
                );
                bundle.f_in_flight.push(
                    dev_ctx
                        .handle
                        .create_fence(&fence_info, None)
                        .context(format!("Failed to create In Flight Fence {}!", i))?,
                );
            };
        }

        Ok(bundle)
    }
}

impl Drop for SyncBundle {
    fn drop(&mut self) {
        unsafe {
            for &s in self.s_image_available.iter() {
                self.dev_ctx.handle.destroy_semaphore(s, None);
            }
            for &s in self.s_render_finished.iter() {
                self.dev_ctx.handle.destroy_semaphore(s, None);
            }
            for &f in self.f_in_flight.iter() {
                self.dev_ctx.handle.destroy_fence(f, None);
            }
        }
    }
}

struct Engine {
    dev_ctx: Arc<DeviceContext>,
    pipeline: PipelineBundle,
    swapchain: SwapchainBundle,
    command: CommandBundle,
    sync: SyncBundle,
    current_frame: usize,
    pub framebuffer_resized: bool,
}

impl Engine {
    pub fn new(window: &Window, required_extensions: &[*const i8]) -> Result<Self> {
        let vk_ctx = Arc::new(VulkanContext::init(window, required_extensions)?);
        let dev_ctx = Arc::new(DeviceContext::init(vk_ctx.clone())?);
        let pipeline = PipelineBundle::new(dev_ctx.clone())?;
        let swapchain = SwapchainBundle::new(dev_ctx.clone(), &pipeline.render_pass)?;
        let command = CommandBundle::new(dev_ctx.clone(), MAX_FRAMES_IN_FLIGHT)?;
        let sync = SyncBundle::new(dev_ctx.clone(), MAX_FRAMES_IN_FLIGHT)?;

        Ok(Engine {
            dev_ctx,
            pipeline,
            swapchain,
            command,
            sync,
            current_frame: 0,
            framebuffer_resized: false,
        })
    }

    pub fn draw_frame(&mut self) -> Result<()> {
        unsafe {
            let wait_fences = [self.sync.f_in_flight[self.current_frame]];
            self.dev_ctx
                .handle
                .wait_for_fences(&wait_fences, true, u64::MAX)
                .context("Failed to wait for In Flight Fence!")?;

            let (image_index, is_suboptimal) = match self.swapchain.loader.acquire_next_image(
                self.swapchain.handle,
                u64::MAX,
                self.sync.s_image_available[self.current_frame],
                vk::Fence::null(),
            ) {
                Ok(result) => result,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => return self.recreate_swapchain(),
                Err(err) => {
                    return Err(err).context("Failed to acquire next image from Swapchain!");
                }
            };

            self.dev_ctx
                .handle
                .reset_fences(&wait_fences)
                .context("Failed to reset In Flight Fence!")?;

            let cmd = [self.record_command(image_index)?];
            let wait_semaphores = [self.sync.s_image_available[self.current_frame]];
            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let signal_semaphores = [self.sync.s_render_finished[self.current_frame]];
            let submit_info = [vk::SubmitInfo::default()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(&cmd)
                .signal_semaphores(&signal_semaphores)];
            self.dev_ctx
                .handle
                .queue_submit(
                    self.dev_ctx.graphics_queue,
                    &submit_info,
                    self.sync.f_in_flight[self.current_frame],
                )
                .context("Failed to submit draw command buffer!")?;

            let swapchains = [self.swapchain.handle];
            let image_indices = [image_index];
            let present_info = vk::PresentInfoKHR::default()
                .wait_semaphores(&signal_semaphores)
                .swapchains(&swapchains)
                .image_indices(&image_indices);
            let changed = match self
                .swapchain
                .loader
                .queue_present(self.dev_ctx.present_queue, &present_info)
            {
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
            self.dev_ctx.handle.device_wait_idle().context("Failed to wait device idle!")?;
        }
        self.swapchain = SwapchainBundle::new(self.dev_ctx.clone(), &self.pipeline.render_pass)?;

        Ok(())
    }

    fn record_command(&self, image_index: u32) -> Result<vk::CommandBuffer> {
        let cmd = self.command.begin_recording(self.current_frame)?;
        let clear_values =
            [vk::ClearValue { color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] } }];
        let render_pass_info = vk::RenderPassBeginInfo::default()
            .render_pass(self.pipeline.render_pass)
            .framebuffer(self.swapchain.framebuffers[image_index as usize])
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.swapchain.extent,
            })
            .clear_values(&clear_values);

        unsafe {
            self.dev_ctx.handle.cmd_begin_render_pass(
                cmd,
                &render_pass_info,
                vk::SubpassContents::INLINE,
            );
            self.dev_ctx.handle.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline.pipeline,
            );
            self.dev_ctx.handle.cmd_set_viewport(
                cmd,
                0,
                &[vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: self.swapchain.extent.width as f32,
                    height: self.swapchain.extent.height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            );
            self.dev_ctx.handle.cmd_set_scissor(
                cmd,
                0,
                &[vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: self.swapchain.extent,
                }],
            );
            self.dev_ctx.handle.cmd_draw(cmd, 3, 1, 0, 0);
            self.dev_ctx.handle.cmd_end_render_pass(cmd);

            self.command.end_recording(cmd)
        }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe {
            self.dev_ctx.handle.device_wait_idle().ok();
        }
    }
}

#[derive(Default)]
struct HelloTriangleApplication {
    state: Option<(Engine, Window)>,
}

impl ApplicationHandler for HelloTriangleApplication {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }

        let result = (|| -> Result<(Engine, Window)> {
            let window = event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Vulkan tutorial with Ash")
                        .with_inner_size(LogicalSize::new(WIDTH, HEIGHT)),
                )
                .context("Failed to create window!")?;
            let required_extensions = ash_window::enumerate_required_extensions(
                window.display_handle().unwrap().as_raw(),
            )
            .context("Failed to enumerate required extensions for surface creation!")?;
            Ok((
                Engine::new(&window, &required_extensions)
                    .context("Failed to initialize Vulkan Engine!")?,
                window,
            ))
        })();

        match result {
            Ok((engine, window)) => {
                self.state = Some((engine, window));
            }
            Err(e) => {
                eprintln!("Failed to initialize application: {}", e);
                event_loop.exit();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some((engine, window)) = self.state.as_mut() else { return };

        match event {
            WindowEvent::CloseRequested => {
                println!("The close button was pressed; stopping");
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                window.request_redraw();
                if let Err(e) = engine.draw_frame() {
                    eprintln!("Failed to draw frame: {}", e);
                }
            }
            WindowEvent::Resized(_size) => {
                engine.framebuffer_resized = true;
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
            println!(
                "[diagnostic] Detected WSL: forcing WINIT_UNIX_BACKEND=x11 and unsetting WAYLAND_DISPLAY"
            );
        }
    }

    let event_loop = EventLoop::new().unwrap();

    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.set_control_flow(ControlFlow::Wait);

    let mut app = HelloTriangleApplication::default();
    let _ = event_loop.run_app(&mut app);
}
