use sdl2;
use lazy_static::{lazy_static};
use sdl2::{pixels::Color, event::Event, keyboard::Keycode, rect::Rect};
use rand::{self, Rng};
use log::debug;
use env_logger;

use std::time;


mod letters;
use letters::{LETTER_SPRITE_HEIGHT, LETTERS};

use std::io::Read;
use std::fs;

use std::collections::HashMap;

const WIDTH:            u32   = 64;
const HEIGHT:           u32   = 32;
const RECT_SIZE:        u32   = 10;
const WINDOW_HEIGHT:    u32   = HEIGHT * RECT_SIZE;
const WINDOW_WIDTH:     u32   = WIDTH  * RECT_SIZE;
const DESIRED_FPS:      u32   = 60;
const FOREGROUND_COLOR: Color = Color::RGB(0xFF, 0xFF, 0xFF);
const BACKGROUND_COLOR: Color = Color::RGB(0x0, 0x0, 0x0);



#[derive(Default, Debug)]
struct Registers {
    pub v: [u8; 16],
    pub i: u16,
    pub dt: u8,
    pub st: u8,
    pub pc: usize,
    pub sp: usize,
}

struct Memory {
    pub stack: [u16; 16],
    pub ram: Vec<u8>,
}

lazy_static! {
    static ref KEYMAP: HashMap<Keycode, usize> = {
        let mut m = HashMap::new();

        m.insert(Keycode::X,    0x0);
        m.insert(Keycode::Num1, 0x1);
        m.insert(Keycode::Num2, 0x2);
        m.insert(Keycode::Num3, 0x3);
        m.insert(Keycode::Q,    0x4);
        m.insert(Keycode::W,    0x5);
        m.insert(Keycode::E,    0x6);
        m.insert(Keycode::A,    0x7);
        m.insert(Keycode::S,    0x8);
        m.insert(Keycode::D,    0x9);
        m.insert(Keycode::Z,    0xA);
        m.insert(Keycode::C,    0xB);
        m.insert(Keycode::Num4, 0xC);
        m.insert(Keycode::R,    0xD);
        m.insert(Keycode::F,    0xE);
        m.insert(Keycode::V,    0xF);
        m
    };
}


struct Keyboard {
    pub pressed: [bool; 16]
}

impl Keyboard {
    pub fn new() -> Self {
        Self {
            pressed: [false; 16]
        }
    }
}


struct Interpreter {
    pub display_updated: bool,
    pub memory: Memory,
    pub registers: Registers,
    pub display: Vec<Vec<bool>>,
    pub keyboard: Keyboard,
    waiting_for_key: Option<usize>,
    rng: rand::rngs::ThreadRng
}

impl Memory {
    pub fn new() -> Self {
        let mut ram = Vec::with_capacity(4096);
        for letter in LETTERS {
            ram.extend_from_slice(&letter);
        }
        ram.extend_from_slice(&[0; 4096 - (16 * LETTER_SPRITE_HEIGHT)]);
        Self {
            stack: Default::default(),
            ram,
        }
    }
}

impl Registers {
    pub fn new() -> Self {
        Self {
            // almost all begin at 512 (0x200), few start at 0x600 but I dont really care honestly
            pc: 512,
            ..Default::default()
        }
    }
}

    impl Interpreter {
        pub fn from_file(path: &str) -> std::io::Result<Self> {
            let mut f = fs::File::open(path)?;
            let mut data = Vec::new();
            f.read_to_end(&mut data)?;
            Ok(Self::new(&data))
        }
        pub fn new(program: &[u8]) -> Self {
            let mut display = Vec::with_capacity(32);
            for _ in 0..32 {
                display.push(vec![false; 64]);
            }
            let mut out = Self {
                display,
                display_updated: true,
                rng: rand::thread_rng(),
                waiting_for_key: None,
                keyboard: Keyboard::new(),
                memory: Memory::new(),
                registers: Registers::new()
            };
            // TODO: this is bad, maybe theres another way to do this
            for (i, e) in program.iter().enumerate() {
                out.memory.ram[512 + i] = *e;
            }
            out
        }
        fn push_stack(&mut self, val: Address) {
            self.registers.sp = self.registers.sp.wrapping_add(1);
            self.memory.stack[self.registers.sp] = val;
        }
        fn pop_stack(&mut self) -> Address {
            let val = self.memory.stack[self.registers.sp];
            self.registers.sp = self.registers.sp.wrapping_sub(1);
            val
        }
        pub fn next(&mut self) {
            if let Some(x) = self.waiting_for_key {
                for (i, pressed) in self.keyboard.pressed.iter().enumerate() {
                    if *pressed {
                        self.registers.v[x] = i as u8;
                        self.waiting_for_key = None;
                    }
                }
                return;
            }
            let next = self.next_instruction();
            self.run_instruction(next);
        }
        fn next_instruction(&mut self) -> Instruction {
            let b1 = self.memory.ram[self.registers.pc];
            let b2 = self.memory.ram[self.registers.pc + 1];
            let raw = u16::from_be_bytes([b1, b2]);
            let instruction = Instruction::from(raw);

        // each instruction is 2 bytes
        self.registers.pc += 2;
        instruction
    }
    fn run_instruction(&mut self, inst: Instruction) {
        let x = inst.x() as usize;
        let y = inst.y() as usize;
        let nibble = inst.nibble();
        let byte = inst.byte();
        let address = inst.address();

        let (a, b, c) = inst.opcode();
        match a {
            // JP addr
            0x1 => {
                debug!("JP 0x{address:x}");
                self.registers.pc = address as usize;
            },
            // CALL addr
            0x2 => {
                debug!("CALL {address:x}");
                self.push_stack(self.registers.pc as u16);
                self.registers.pc = address as usize;
            },
            // SE Vx, byte
            0x3 => {
                debug!("SE V{x:x}, 0x{byte:x}");
                if self.registers.v[x] == byte {
                    _ = self.next_instruction();
                }
            },
            // SNE Vx, byte
            0x4 => {
                debug!("SNE V{x:x}, 0x{byte:x}");
                if self.registers.v[x] != byte {
                    _ = self.next_instruction(); }
            },
            // LD Vx, byte
            0x6 => {
                debug!("LD V{x:x}, 0x{byte:x}");
                self.registers.v[x] = byte;
            },
            // ADD Vx, byte
            0x7 => {
                debug!("ADD V{x:x}, 0x{byte:x}");
                self.registers.v[x] = self.registers.v[x].wrapping_add(byte);
            },
            // LD I, addr
            0xA => {
                debug!("LD I, 0x{address:x}");
                self.registers.i = address;
            },
            // JP V0, addr
            0xB => {
                debug!("JP V0, 0x{address:x}");
                self.registers.pc = (self.registers.v[0] as u16 + address) as usize;
            },
            // RND Vx, byte
            0xC => {
                debug!("RND V{x:x}, 0x{byte:x}");
                self.registers.v[x] = self.rng.gen_range(0..=255) & byte;
            },
            // DRW Vx, Vy, nibble
            0xD => {
                debug!("DRW V{x:x}, V{y:x}, 0x{nibble:x}");
                self.display_updated = true;
                let ri = self.registers.i as usize;
                let x = (self.registers.v[x] % WIDTH as u8) as usize;
                let y = (self.registers.v[y] % HEIGHT as u8) as usize;
                let mut overwrite = false;

                for i in 0..nibble as usize {
                    let byte = self.memory.ram[ri + i];
                    // because MSB
                    for q in 0..8 as usize {
                        let bit = ((byte >> q.abs_diff(7)) & 1) != 0;
                        if !bit {
                            continue;
                        }
                        if let Some(u) = self.display.get_mut(y + i) {
                            if let Some(handle) = u.get_mut(x + q) {
                                match *handle {
                                    true => {
                                        *handle = false;
                                        overwrite = true;
                                    },
                                    false => *handle = true
                                }
                            }
                        }
                    }
                }
                self.registers.v[0xF] = if overwrite { 1 } else { 0 };
            },
            _ => match (a, c) {
                // SE Vx, Vy
                (0x5, 0x0) => {
                    debug!("SE V{x:x}, V{y:x}");
                    if self.registers.v[x] == self.registers.v[y] {
                        _ = self.next_instruction();
                    }
                },
                // LD Vx, Vy
                (0x8, 0x0) => {
                    debug!("LD V{x:x}, V{y:x}");
                    self.registers.v[x] = self.registers.v[y];
                }
                // OR Vx, Vy
                (0x8, 0x1) => {
                    debug!("OR V{x:x}, V{y:x}");
                    self.registers.v[x] |= self.registers.v[y]; 
                }
                // AND Vx, Vy
                (0x8, 0x2) => {
                    debug!("AND V{x:x}, V{y:x}");
                    self.registers.v[x] &= self.registers.v[y]; 
                }
                // XOR Vx, Vy
                (0x8, 0x3) => {
                    debug!("XOR V{x:x}, V{y:x}");
                    self.registers.v[x] ^= self.registers.v[y]; 
                }
                // ADD Vx, VY
                (0x8, 0x4) => {
                    debug!("ADD V{x:x}, V{y:x}");
                    let prev_x = self.registers.v[x];
                    self.registers.v[x] = self.registers.v[x].wrapping_add(self.registers.v[y]);
                    self.registers.v[0xF] = 
                        if prev_x > self.registers.v[x] { 0x1 } else { 0x0 };
                }
                // SUB Vx, Vy
                (0x8, 0x5) => {
                    let prev_x = self.registers.v[x];
                    debug!("SUB V{x:x}, V{y:x}");
                    self.registers.v[x] = self.registers.v[x].wrapping_sub(self.registers.v[y]);
                    self.registers.v[0xF] = if prev_x > self.registers.v[y] { 1 } else { 0 };
                }
                // SHR Vx, {, Vy}
                (0x8, 0x6) => {
                    debug!("SHR V{x:x}, V{y:x}");
                    let prev_x = self.registers.v[x];
                    self.registers.v[x] = self.registers.v[x].wrapping_div(2);
                    self.registers.v[0xF] = if prev_x & 0x01 > 0 {1} else {0};
                },
                // SUBN Vx, VY
                (0x8, 0x7) => {
                    debug!("SUBN V{x:x}, V{y:x}");
                    let prev_x = self.registers.v[x];
                    self.registers.v[x] = self.registers.v[y].wrapping_sub(self.registers.v[x]);
                    self.registers.v[0xF] = if self.registers.v[y] > prev_x { 1 } else { 0 };
                },
                // SHL Vx {, Vy}
                (0x8, 0xE) => {
                    debug!("SHL V{x:x}, V{y:x}");
                    let prev_x = self.registers.v[x];
                    self.registers.v[x] = self.registers.v[x].wrapping_mul(2);
                    self.registers.v[0xF] = if prev_x & 0b10000000 != 0 { 1 } else { 0 };
                },
                // SNE Vx, Vy
                (0x9, 0x0) => {
                    debug!("SNE V{x:x}, V{y:x}");
                    if self.registers.v[x] != self.registers.v[y] {
                        _ = self.next_instruction();
                    }
                },
                _ => match (a, b, c) {
                    // CLS
                    (0x0, 0xE, 0x0) => {
                        debug!("CLS");
                        self.display_updated = true;
                        for col in &mut self.display {
                            for item in col {
                                *item = false;
                            }
                        }
                    },
                    // RET
                    (0x0, 0xE, 0xE) => {
                        debug!("RET");
                        self.registers.pc = self.pop_stack() as usize;
                    },
                    // SYS addr (does nothing)
                    (0x0, _, _) => {
                        debug!("SYS 0x{address:x}");
                    },
                    // SKP Vx
                    (0xE, 0x9, 0xE) => {
                        debug!("SKP V{x:x}");
                        if self.keyboard.pressed[self.registers.v[x] as usize] {
                            _ = self.next_instruction();
                        }

                    },
                    // SKNP Vx
                    (0xE, 0xA, 0x1) => {
                        debug!("SKNP V{x:x}");
                        if !self.keyboard.pressed[self.registers.v[x] as usize] {
                            _ = self.next_instruction();
                        }
                    },
                    // LD Vx, DT
                    (0xF, 0x0, 0x7) => {
                        debug!("LD V{x:x}, DT");
                        self.registers.v[x] = self.registers.dt;
                    },
                    // LD Vx, K
                    (0xF, 0x0, 0xA) => {
                        debug!("LD V{x:x}, K");
                        self.waiting_for_key = Some(x);
                    },
                    // LD DT, Vx
                    (0xF, 0x1, 0x5) => {
                        debug!("LD DT, V{x:x}");
                        self.registers.dt = self.registers.v[x];
                    },
                    // LD ST, Vx
                    (0xF, 0x1, 0x8) => {
                        debug!("LD ST, V{x:x}");
                        self.registers.st = self.registers.v[x];
                    },
                    // ADD I, Vx
                    (0xF, 0x1, 0xE) => {
                        debug!("ADD I, V{x:x}");
                        self.registers.i += self.registers.v[x] as u16;
                    },
                    // LD F, Vx
                    (0xF, 0x2, 0x9) => {
                        debug!("LD F, V{x:x}");
                        self.registers.i = (LETTER_SPRITE_HEIGHT * x) as u16;
                    },
                    // LD B, Vx
                    (0xF, 0x3, 0x3) => {
                        debug!("LD B, V{x:x}");
                        let mut n = self.registers.v[x];
                        let i = self.registers.i as usize;

                        let digits = { 
                            let mut digits = Vec::with_capacity(3);
                            while n > 9 {
                                digits.push(n % 10);
                                n /= 10;
                            }
                            digits.push(n);
                            digits
                        };

                        self.memory.ram[i] = *digits.get(2).unwrap_or(&0);
                        self.memory.ram[i + 1] = *digits.get(1).unwrap_or(&0);
                        self.memory.ram[i + 2] = digits[0];

                    },
                    // LD [I], Vx
                    (0xF, 0x5, 0x5) => {
                        debug!("LD [I], V{x:x}");
                        let ri = self.registers.i as usize;
                        for i in 0..=x {
                            self.memory.ram[ri + i] = self.registers.v[i];
                        }
                    },
                    // LD Vx, [I]
                    (0xF, 0x6, 0x5) => {
                        debug!("LD [Vx, [I]");
                        let ri = self.registers.i as usize;
                        for i in 0..=x {
                            self.registers.v[i] = self.memory.ram[ri + i];
                        }
                    },
                    _ => panic!("invalid instruction: {:?}", inst)
                }
            }
        }
    }
}

type Address = u16;
type Byte = u8;
type Nibble = u8;

#[derive(Debug)]
struct Instruction {
    data: [u8; 4]
}

impl From<u16> for Instruction {
    fn from(value: u16) -> Self {
        let [high, low] = value.to_be_bytes();
        let data = [
            (high & 0xF0) >> 0x4,
            (high & 0x0F),
            (low  & 0xF0) >> 0x4,
            (low  & 0x0F)
        ];
        Instruction {
            data
        }
    }
}

impl Instruction {
    pub fn x(&self) -> Nibble {
        self.data[1]
    }
    pub fn y(&self) -> Nibble {
        self.data[2]
    }
    pub fn address(&self) -> Address {
        self.data[1..].iter().fold(0, |x, &i| x << 4 | i as u16)
    }
    pub fn byte(&self) -> Byte {
        (self.data[2] << 0x4) + self.data[3]
    }
    pub fn nibble(&self) -> Nibble {
        self.data[3]
    }
    pub fn opcode(&self) -> (Nibble, Nibble, Nibble) {
        (self.data[0], self.data[2], self.data[3])
    }
}

#[cfg(test)]
mod tests {
    use crate::*;
    #[test]
    fn nibbling() {
        let instruction = Instruction::from(0xABCD);
        assert_eq!(instruction.address(), 0xBCD);
        assert_eq!(instruction.x(), 0xB);
        assert_eq!(instruction.y(), 0xC);
        assert_eq!(instruction.byte(), 0xCD);
    }
    #[test]
    fn some_instructions() {
        let program = [
            0x60, 0x05, // LD V0 0x5
            0xA2, 0x58, // LD I, 0x258
            0xF3, 0x55, // LD [I], Vx
            0x60, 0x05, //  LD V0, 0x5
            0x80, 0x1E,  // SHL V0, V1
            0x60, 0xFF, //  LD V0, 0x5
            0x80, 0x1E  // SHL V0, V1
        ];
        let mut interpreter = Interpreter::new(&program);
        interpreter.next();
        //:assert_eq!(0x5, interpreter.registers.v[0]);
        interpreter.next();
        assert_eq!(0x258, interpreter.registers.i);
        interpreter.next();
        let i = interpreter.registers.i as usize;
        assert_eq!(&[0x05, 0x0, 0x0], &interpreter.memory.ram[i..i+3]);
        interpreter.next();
        interpreter.next();
        assert_eq!(interpreter.registers.v[0], 10);
        assert_eq!(interpreter.registers.v[0xF], 0);
        interpreter.next();
        interpreter.next();
        assert_eq!(interpreter.registers.v[0], 254);
        assert_eq!(interpreter.registers.v[0xF], 1);
    }

    #[test]
    fn draw() {
        let program = [
            0xF0, 0x29, // LD F, V0
            0xF0, 0x55, // LD [I], V0
            0x60, 0x10, // LD V0 0x10
            0x61, 0x10, // LD V1 0x10
            0xD0, 0x15, // DRW V0 V1 0x5
            0xD0, 0x15, // DRW V0 V1 0x5
        ];
        let mut interpreter = Interpreter::new(&program);
        interpreter.next();
        interpreter.next();
        interpreter.next();
        interpreter.next();
        interpreter.next();
        interpreter.next();
        assert_eq!(interpreter.registers.v[0xF], 1);
    }
}


fn main() -> std::io::Result<()> {
    env_logger::init();

    let mut interpreter = if let Some(arg) = std::env::args().skip(1).next() {
        Interpreter::from_file(&arg)?
    } else {
        panic!("you need to supply a file to run!");
    };
    let sdl_ctx = sdl2::init().unwrap();
    let video_subsystem = sdl_ctx.video().unwrap();

    let window = video_subsystem.window("uwu", WINDOW_WIDTH, WINDOW_HEIGHT)
        .position_centered()
        .build() .unwrap(); let mut canvas = window.into_canvas().build().unwrap();
    let mut event_pump = sdl_ctx.event_pump().unwrap();
    let mut quit = false;
    let mut last_time = time::Instant::now();
    while !quit {
        if last_time.elapsed().as_millis() < 200 / DESIRED_FPS as u128 {
            continue
        }
        last_time = time::Instant::now();

        if interpreter.registers.dt > 0 {
            interpreter.registers.dt -= 1;
        }
        if interpreter.registers.st > 0 {
            interpreter.registers.st -= 1;
        }

        interpreter.next();

        for event in event_pump.poll_iter() {
            match event {
                Event::Quit {..} |
                    Event::KeyDown { keycode: Some(Keycode::Escape), .. } => quit = true,
                Event::KeyDown { keycode: Some(Keycode::Space), ..} => {
                    interpreter.next();
                    println!("DELAY TIMER: {}", interpreter.registers.dt);
                }
                Event::KeyDown { keycode: Some(key), .. } => {
                    if let Some(index) = KEYMAP.get(&key) {
                        interpreter.keyboard.pressed[*index] = true;
                    }
                },
                Event::KeyUp { keycode: Some(key), .. } => {
                    if let Some(index) = KEYMAP.get(&key) {
                        interpreter.keyboard.pressed[*index] = false;
                    }
                },
                _ => {}
            }
        }

        if !interpreter.display_updated {
            continue;
        }

        interpreter.display_updated = false;

        canvas.set_draw_color(BACKGROUND_COLOR);
        canvas.clear();
        canvas.set_draw_color(FOREGROUND_COLOR);

        for (i, collumn) in interpreter.display.clone().into_iter().enumerate() {
            for (j, row) in collumn.into_iter().enumerate() {
                if !row { continue };
                let rect = Rect::new(
                    (j * RECT_SIZE as usize) as i32,
                    (i * RECT_SIZE as usize) as i32,
                    RECT_SIZE,
                    RECT_SIZE
                );
                canvas.fill_rect(Some(rect)).unwrap();
            }
        }
        canvas.present();
    }
    Ok(())
}
