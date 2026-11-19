/// Applies `CREATE_NO_WINDOW` (0x08000000) on Windows so child processes
/// (yt-dlp, ffmpeg, ffprobe, whisper-cli, powershell, nvidia-smi) never flash
/// console windows or block under `windows_subsystem = "windows"`.
pub fn configure_std_command(cmd: &mut std::process::Command) -> &mut std::process::Command {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x08000000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    cmd
}

pub fn configure_tokio_command(cmd: &mut tokio::process::Command) -> &mut tokio::process::Command {
    #[cfg(windows)]
    {
        const CREATE_NO_WINDOW: u32 = 0x08000000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    cmd
}
