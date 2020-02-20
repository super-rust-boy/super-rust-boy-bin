# Super Rust Boy binary
Uses the super-rust-boy library to run the emulator.

### Options
Run with `-p=g` for classic green palette, `-p=bw` for greyscale palette. `-m` to mute. `-s "SAVE_FILE_NAME"` to specify a save file. By default a save file will be created (if needed) with the name of the cart in the same directory, with `.sav` extension.

By default the emulator will try and run the game in colour if it is available. Certain original Game Boy games have unique palettes that will be selected (made by Nintendo). If you don't want these, force the palette with `-p=g` for classic green palette, `-p=bw` for greyscale palette. Running a Game Boy Color game that is compatible with the original Game Boy (such as Pokemon Gold/Silver) will run in classic mode if you set one of these palettes.

### Example:
To run a ROM in debug mode:
`cargo run --release -- path/to/ROM.gb --debug`

To run a ROM with a custom save file and using a greyscale palette:
`cargo run --release -- path/to/ROM.gb -s=path/to/save_file.sav -p=bw`