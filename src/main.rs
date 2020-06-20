extern crate rustboy;

#[cfg(feature = "debug")]
mod debug;

mod shaders;

use rustboy::*;

use cpal;
use cpal::traits::{
    HostTrait,
    DeviceTrait,
    EventLoopTrait
};
use clap::{clap_app, crate_version};
use chrono::Utc;
use winit::{
    EventsLoop,
    Event,
    WindowEvent,
    WindowBuilder,
    ElementState,
    VirtualKeyCode
};

use vulkano::{
    instance::{
        Instance, PhysicalDevice
    },
    device::{
        Device, DeviceExtensions
    },
    framebuffer::{
        Framebuffer, Subpass, FramebufferAbstract, RenderPassAbstract
    },
    pipeline::{
        GraphicsPipeline,
        viewport::Viewport
    },
    command_buffer::{
        AutoCommandBufferBuilder,
        DynamicState
    },
    sampler::{
        Filter,
        MipmapMode,
        Sampler,
        SamplerAddressMode
    },
    swapchain::{
        Swapchain, SurfaceTransform, PresentMode, acquire_next_image, CompositeAlpha
    },
    sync::{
        now, GpuFuture
    },
    descriptor::{
        descriptor_set::FixedSizeDescriptorSetsPool,
    },
    buffer::{
        BufferUsage,
        ImmutableBuffer
    },
    image::{
        Dimensions,
        immutable::ImmutableImage
    },
    format::Format
};

use vulkano_win::VkSurfaceBuild;
use std::sync::Arc;


const FRAME_TIME: i64 = 16_666;
//const FRAME_TIME: i64 = 16_743; // 59.73 fps

#[derive(Default, Debug, Clone)]
struct Vertex {
    position:   [f32; 2],
    tex_coord:  [f32; 2]
}

vulkano::impl_vertex!(Vertex, position, tex_coord);

fn main() {
    let app = clap_app!(rustboy =>
        (version: crate_version!())
        (author: "Simon Cooper")
        (about: "Game Boy and Game Boy Color emulator.")
        (@arg CART: "The path to the game cart to use.")
        (@arg debug: -d "Enter debug mode.")
        (@arg mute: -m "Mutes the emulator.")
        (@arg palette: -p +takes_value "Choose a palette. 'g' selects the classic green scheme, 'bw' forces greyscale. By default SGB colour will be used if available.")
        (@arg save: -s +takes_value "Save file path.")
    );

    let cmd_args = app.get_matches();

    let cart = match cmd_args.value_of("CART") {
        Some(c) => c.to_string(),
        None => panic!("Usage: rustboy [cart name]. Run with --help for more options."),
    };

    let save_file = match cmd_args.value_of("save") {
        Some(c) => c.to_string(),
        None => make_save_name(&cart),
    };

    let palette = choose_palette(cmd_args.value_of("palette"));

    // Video
    let mut events_loop = EventsLoop::new();

    let rom_type = ROMType::File(cart);
    let mut rustboy = RustBoy::new(rom_type, &save_file, palette);

    //let mut averager = avg::Averager::<i64>::new(60);
    let mut frame_tex = [255_u8; 160 * 144 * 4];

    if cmd_args.is_present("debug") {
        #[cfg(feature = "debug")]
        debug::debug_mode(&mut rustboy);
    } else {
        if !cmd_args.is_present("mute") {
            run_audio(&mut rustboy);
        }

        // Make instance with window extensions.
        let instance = {
            let extensions = vulkano_win::required_extensions();
            Instance::new(None, &extensions, None).expect("Failed to create vulkan instance")
        };

        // Get graphics device.
        let physical = PhysicalDevice::enumerate(&instance).next()
            .expect("No device available");

        // Get graphics command queue family from graphics device.
        let queue_family = physical.queue_families()
            .find(|&q| q.supports_graphics())
            .expect("Could not find a graphical queue family");

        // Make software device and queue iterator of the graphics family.
        let (device, mut queues) = {
            let device_ext = DeviceExtensions{
                khr_swapchain: true,
                .. DeviceExtensions::none()
            };
            
            Device::new(physical, physical.supported_features(), &device_ext,
                        [(queue_family, 0.5)].iter().cloned())
                .expect("Failed to create device")
        };

        // Get a queue from the iterator.
        let queue = queues.next().unwrap();

        // Make a surface.
        let surface = WindowBuilder::new()
            .with_dimensions((320, 288).into())
            .with_title("Super Rust Boy")
            .build_vk_surface(&events_loop, instance.clone())
            .expect("Couldn't create surface");

        // Make the sampler for the texture.
        let sampler = Sampler::new(
            device.clone(),
            Filter::Nearest,
            Filter::Nearest,
            MipmapMode::Nearest,
            SamplerAddressMode::Repeat,
            SamplerAddressMode::Repeat,
            SamplerAddressMode::Repeat,
            0.0, 1.0, 0.0, 0.0
        ).expect("Couldn't create sampler!");

        // Get a swapchain and images for use with the swapchain, as well as the dynamic state.
        let ((swapchain, images), dynamic_state) = {

            let caps = surface.capabilities(physical)
                    .expect("Failed to get surface capabilities");
            let dimensions = caps.current_extent.unwrap_or([160, 144]);

            //let alpha = caps.supported_composite_alpha.iter().next().unwrap();
            //println!("{:?}", caps.supported_formats);
            let format = caps.supported_formats[0].0;

            (Swapchain::new(device.clone(), surface.clone(),
                caps.min_image_count, format, dimensions, 1, caps.supported_usage_flags, &queue,
                SurfaceTransform::Identity, CompositeAlpha::Opaque, PresentMode::Fifo, true, None
            ).expect("Failed to create swapchain"),
            DynamicState {
                viewports: Some(vec![Viewport {
                    origin: [0.0, 0.0],
                    dimensions: [dimensions[0] as f32, dimensions[1] as f32],
                    depth_range: 0.0 .. 1.0,
                }]),
                .. DynamicState::none()
            })
        };

        // Make the render pass to insert into the command queue.
        let render_pass = Arc::new(vulkano::single_pass_renderpass!(device.clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: swapchain.format(),//Format::R8G8B8A8Unorm,
                    samples: 1,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {}
            }
        ).unwrap()) as Arc<dyn RenderPassAbstract + Send + Sync>;

        let framebuffers = images.iter().map(|image| {
            Arc::new(
                Framebuffer::start(render_pass.clone())
                    .add(image.clone()).unwrap()
                    .build().unwrap()
            ) as Arc<dyn FramebufferAbstract + Send + Sync>
        }).collect::<Vec<_>>();

        // Assemble
        let vs = shaders::vs::Shader::load(device.clone()).expect("failed to create vertex shader");
        let fs = shaders::fs::Shader::load(device.clone()).expect("failed to create fragment shader");

        // Make pipeline.
        let pipeline = Arc::new(GraphicsPipeline::start()
            .vertex_input_single_buffer::<Vertex>()
            .vertex_shader(vs.main_entry_point(), ())
            .triangle_strip()
            .viewports_dynamic_scissors_irrelevant(1)
            .fragment_shader(fs.main_entry_point(), ())
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device.clone())
            .unwrap()
        );

        // Make descriptor set pools.
        let mut set_pool = FixedSizeDescriptorSetsPool::new(pipeline.clone(), 0);

        let (vertices, vertex_future) = ImmutableBuffer::from_iter(
            vec![
                Vertex{position: [-1.0, -1.0], tex_coord: [0.0, 0.0]},
                Vertex{position: [1.0, -1.0], tex_coord: [1.0, 0.0]},
                Vertex{position: [-1.0, 1.0], tex_coord: [0.0, 1.0]},
                Vertex{position: [1.0, 1.0], tex_coord: [1.0, 1.0]},
            ].into_iter(),
            BufferUsage::vertex_buffer(),
            queue.clone()
        ).unwrap();

        let mut previous_frame_future = Box::new(vertex_future) as Box<dyn GpuFuture>;

        loop {
            //println!("Frame");
            let frame = Utc::now();

            read_inputs(&mut events_loop, &mut rustboy);
            rustboy.frame(&mut frame_tex);

            // Get current framebuffer index from the swapchain.
            let (image_num, acquire_future) = acquire_next_image(swapchain.clone(), None).expect("Didn't get next image");

            // Get image with current texture.
            let (image, image_future) = ImmutableImage::from_iter(
                frame_tex.iter().cloned(),
                Dimensions::Dim2d { width: 160, height: 144 },
                Format::R8G8B8A8Uint,
                queue.clone()
            ).expect("Couldn't create image.");

            // Make descriptor set to bind texture.
            let set0 = Arc::new(set_pool.next()
                .add_sampled_image(image, sampler.clone()).unwrap()
                .build().unwrap());

            // Start building command buffer using pipeline and framebuffer, starting with the background vertices.
            let command_buffer = AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family()).unwrap()
                .begin_render_pass(framebuffers[image_num].clone(), false, vec![[1.0, 0.0, 0.0, 1.0].into()]).unwrap()
                .draw(
                    pipeline.clone(),
                    &dynamic_state,
                    vertices.clone(),
                    set0.clone(),
                    ()
                ).unwrap().end_render_pass().unwrap().build().unwrap();

            // Wait until previous frame is done.
            let mut now_future = Box::new(now(device.clone())) as Box<dyn GpuFuture>;
            std::mem::swap(&mut previous_frame_future, &mut now_future);

            // Wait until previous frame is done,
            // _and_ the framebuffer has been acquired,
            // _and_ the texture has been uploaded.
            let future = now_future.join(acquire_future)
                .join(image_future)
                .then_execute(queue.clone(), command_buffer).unwrap()                   // Run the commands (pipeline and render)
                .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)    // Present newly rendered image.
                .then_signal_fence_and_flush();                                         // Signal done and flush the pipeline.

            match future {
                Ok(future) => previous_frame_future = Box::new(future) as Box<_>,
                Err(e) => println!("Err: {:?}", e),
            }

            previous_frame_future.cleanup_finished();

            //averager.add((Utc::now() - frame).num_milliseconds());
            //println!("Frame t: {}ms", averager.get_avg());

            while (Utc::now() - frame) < chrono::Duration::microseconds(FRAME_TIME) {}  // Wait until next frame.
        }
    }
}

fn make_save_name(cart_name: &str) -> String {
    match cart_name.find(".") {
        Some(pos) => cart_name[0..pos].to_string() + ".sav",
        None      => cart_name.to_string() + ".sav"
    }
}

fn choose_palette(palette: Option<&str>) -> UserPalette {
    match palette {
        Some(s) => match s {
            "g" => UserPalette::Classic,
            "bw" => UserPalette::Greyscale,
            _ => UserPalette::Default
        },
        None => UserPalette::Default
    }
}

fn read_inputs(events_loop: &mut EventsLoop, rustboy: &mut RustBoy) {
    events_loop.poll_events(|e| {
        match e {
            Event::WindowEvent {
                window_id: _,
                event: w,
            } => match w {
                WindowEvent::CloseRequested => {
                    ::std::process::exit(0);
                },
                WindowEvent::KeyboardInput {
                    device_id: _,
                    input: k,
                } => {
                    let pressed = match k.state {
                        ElementState::Pressed => true,
                        ElementState::Released => false,
                    };
                    match k.virtual_keycode {
                        Some(VirtualKeyCode::X)         => rustboy.set_button(Button::A, pressed),
                        Some(VirtualKeyCode::Z)         => rustboy.set_button(Button::B, pressed),
                        Some(VirtualKeyCode::Space)     => rustboy.set_button(Button::Select, pressed),
                        Some(VirtualKeyCode::Return)    => rustboy.set_button(Button::Start, pressed),
                        Some(VirtualKeyCode::Up)        => rustboy.set_button(Button::Up, pressed),
                        Some(VirtualKeyCode::Down)      => rustboy.set_button(Button::Down, pressed),
                        Some(VirtualKeyCode::Left)      => rustboy.set_button(Button::Left, pressed),
                        Some(VirtualKeyCode::Right)     => rustboy.set_button(Button::Right, pressed),
                        _ => {},
                    }
                },
                //WindowEvent::Resized(_) => rustboy.on_resize(),
                _ => {}
            },
            _ => {},
        }
    });
}

fn run_audio(rustboy: &mut RustBoy) {
    let host = cpal::default_host();
    let event_loop = host.event_loop();
    let device = host.default_output_device().expect("no output device available.");
    let mut supported_formats_range = device.supported_output_formats()
        .expect("error while querying formats");
    let format = supported_formats_range.next()
        .expect("No supported format")
        .with_max_sample_rate();
    let stream_id = event_loop.build_output_stream(&device, &format).unwrap();
    let sample_rate = format.sample_rate.0 as usize;

    let mut audio_handler = rustboy.enable_audio(sample_rate);
    std::thread::spawn(move || {
        event_loop.play_stream(stream_id).expect("Stream could not start.");

        event_loop.run(move |_stream_id, stream_result| {
            use cpal::StreamData::*;
            use cpal::UnknownTypeOutputBuffer::*;

            let stream_data = match stream_result {
                Ok(data) => data,
                Err(e) => {
                    eprintln!("An error occurred in audio handler: {}", e);
                    return;
                }
            };

            match stream_data {
                Output { buffer: U16(mut buffer) } => {
                    let mut packet = vec![0.0; buffer.len()];
                    audio_handler.get_audio_packet(&mut packet);
                    for (out, p) in buffer.chunks_exact_mut(2).zip(packet.chunks_exact(2)) {
                        for (elem, f) in out.iter_mut().zip(p.iter()) {
                            *elem = (f * u16::max_value() as f32) as u16
                        }
                    }
                },
                Output { buffer: I16(mut buffer) } => {
                    let mut packet = vec![0.0; buffer.len()];
                    audio_handler.get_audio_packet(&mut packet);
                    for (out, p) in buffer.chunks_exact_mut(2).zip(packet.chunks_exact(2)) {
                        for (elem, f) in out.iter_mut().zip(p.iter()) {
                            *elem = (f * i16::max_value() as f32) as i16
                        }
                    }
                },
                Output { buffer: F32(mut buffer) } => {
                    audio_handler.get_audio_packet(&mut buffer);
                },
                _ => {},
            }
        });
    });
}

/*
// Averager

use std::{
    collections::VecDeque,
    ops::{
        Add,
        Div
    }
};

pub struct Averager<T: Add + Div> {
    queue:      VecDeque<T>,
    max_len:    usize
}

impl Averager<i64> {
    pub fn new(len: usize) -> Self {
        Averager {
            queue:      VecDeque::with_capacity(len),
            max_len:    len
        }
    }

    pub fn add(&mut self, data: i64) {
        self.queue.push_back(data);

        if self.queue.len() > self.max_len {
            self.queue.pop_front();
        }
    }

    pub fn get_avg(&self) -> i64 {
        self.queue.iter().sum::<i64>() / (self.queue.len() as i64)
    }
}
*/