use nokhwa::{
    pixel_format::RgbFormat,
    utils::{CameraIndex, RequestedFormat, RequestedFormatType},
    Camera,
};
use std::time::Instant;

const ZENOH_KEY: &str = "kornia/gpu/frame";
const DST_W: u32 = 1280;
const DST_H: u32 = 720;
const JPEG_QUALITY: i32 = 80;

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let _use_cuda = args.iter().any(|a| a == "--cuda");

    let backend = kornia_gpu::auto_select_backend();
    println!("[kornia-gpu] backend: {}", backend.name());
    println!("[bubbaloop] GPU backend: {}", backend.name());

    let mut camera = Camera::new(
        CameraIndex::Index(0),
        RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate),
    )
    .expect("failed to open webcam");
    camera.open_stream().expect("failed to open camera stream");

    let fmt = camera.camera_format();
    let src_w = fmt.width();
    let src_h = fmt.height();
    println!(
        "[bubbaloop] camera: {}x{} @ {}fps",
        src_w,
        src_h,
        fmt.frame_rate()
    );

    // Build streaming pipeline once - persistent buffers, 1-frame latency.
    // push_frame returns None on the first call (pipeline filling).
    let mut pipeline = backend
        .streaming_pipeline(src_w, src_h)
        .expect("backend does not support streaming pipeline");

    let session = zenoh::open(zenoh::Config::default())
        .await
        .expect("failed to open zenoh session");
    let publisher = session
        .declare_publisher(ZENOH_KEY)
        .await
        .expect("failed to declare zenoh publisher");

    println!("[bubbaloop] publishing to zenoh key: {ZENOH_KEY}");
    println!("[bubbaloop] pipeline: RGB → GpuStreamingPipeline (gray_from_rgb) → JPEG → Zenoh");
    println!("[bubbaloop] press Ctrl-C to stop\n");

    let mut frame_idx: u64 = 0;

    loop {
        let t_total = Instant::now();

        // capture
        let raw_frame = camera.frame().expect("failed to capture frame");
        let decoded = raw_frame
            .decode_image::<RgbFormat>()
            .expect("failed to decode frame");
        let rgb_bytes: &[u8] = decoded.as_raw();

        // GPU - push into streaming pipeline; returns previous frame's result
        // Returns None on frame 0 (pipeline filling), Some(gray bytes) after
        let t_kernel = Instant::now();
        let gpu_result = pipeline.push_frame(rgb_bytes, src_w, src_h);
        let kernel_ms = t_kernel.elapsed().as_secs_f64() * 1000.0;

        let Some(gray_bytes) = gpu_result else {
            frame_idx += 1;
            continue; // first frame - pipeline filling, nothing to publish yet
        };

        // JPEG encode via turbojpeg (libjpeg-turbo) - ~5ms vs ~88ms for image crate
        let t_encode = Instant::now();
        let jpeg_bytes = turbojpeg::compress(
            turbojpeg::Image {
                pixels: gray_bytes.as_slice(),
                width: DST_W as usize,
                pitch: DST_W as usize, // bytes per row for grayscale = width
                height: DST_H as usize,
                format: turbojpeg::PixelFormat::GRAY,
            },
            JPEG_QUALITY,
            turbojpeg::Subsamp::Gray,
        )
        .expect("JPEG encode failed");
        let encode_ms = t_encode.elapsed().as_secs_f64() * 1000.0;

        // publish
        publisher
            .put(jpeg_bytes.as_ref()) // OwnedBuf implements AsRef<[u8]>
            .await
            .expect("zenoh put failed");

        let total_ms = t_total.elapsed().as_secs_f64() * 1000.0;

        println!(
            "frame {:>6} | src {}x{} | kernel {:>6.2}ms | encode {:>6.2}ms | total {:>7.2}ms | jpeg {}B",
            frame_idx,
            src_w,
            src_h,
            kernel_ms,
            encode_ms,
            total_ms,
            jpeg_bytes.len(),
        );

        frame_idx += 1;
    }
}
