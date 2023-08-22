/// Block data, whose size is known by the Kernel
#[derive(Clone, Debug, Default)]
pub struct Block(Box<[bool]>);

pub trait Kernel {
    /// Power law size of the basic block. E.g. each block has a width of 2^n, where n = self.order()
    /// For HashLife/Conway life (radius 1), this would be 2, since input would be 4 2x2 blocks and output would be 1 2x2 block
    /// For an MNCA with radius 4, this would be 3, since the input would be 4 8x8 blocks and output would be 1 8x8 block
    fn order(&self) -> usize;

    /// Given a novel combination of 4 blocks, produce an output block advanced by one time step
    /// (each entry in the input and output blocks are either 0 or 1 indicating dead or live states respectively)
    fn approximate(&mut self, blocks: [Block; 4]) -> (Block, KernelResult);
}

#[derive(Clone, Debug, Copy)]
pub enum KernelResult {
    /// An exact, new block has been returned. This new block should be hashed and inserted into the macrocell tree.
    NewBlock,
    /// An approximate block has been returned. This new block should be hashed and should be found to already be in the macrocell tree.
    Approximate,
}

struct Dense {
    blocks: Vec<Block>,
    width: usize,
    kernel: Box<dyn Kernel>,
}

impl Dense {
    pub fn new(kernel: Box<dyn Kernel>, width: usize, height: usize) -> Self {
        Self {
            blocks: vec![Block::zero(&*kernel); width * height],
            width,
            kernel,
        }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        assert_eq!(
            self.blocks.len() % self.width(),
            0,
            "Invalid number of blocks..."
        );
        self.blocks.len() / self.width()
    }

    pub fn index(&self, index: (i32, i32)) -> Option<usize> {
        let (x, y) = index;
        let in_bounds = x >= 0 && y >= 0 && x < self.width() as i32 && y < self.height() as i32;
        in_bounds.then(|| x as usize + y as usize * self.height())
    }

    pub fn get(&self, index: (i32, i32)) -> Option<&Block> {
        self.index(index).and_then(|i| self.blocks.get(i))
    }

    pub fn get_mut(&mut self, index: (i32, i32)) -> Option<&mut Block> {
        self.index(index).and_then(|i| self.blocks.get_mut(i))
    }

    pub fn get_pixel(&self, pixel: (i32, i32)) -> bool {
        let (x, y) = pixel;
        let x_block = x / self.width() as i32;
        let y_block = x / self.height() as i32;
        let block = &self.get((x_block, y_block)).unwrap();

        block.get(&*self.kernel, (x, y))
    }
}

impl Block {
    pub fn zero(ker: &dyn Kernel) -> Self {
        Self::new(ker, vec![false; Self::width(ker).pow(2)])
    }

    pub fn new(ker: &dyn Kernel, data: Vec<bool>) -> Self {
        let expected_len = Self::width(ker).pow(2);
        assert_eq!(expected_len, data.len());
        Self(data.into_boxed_slice())
    }

    pub fn width(ker: &dyn Kernel) -> usize {
        1 << ker.order()
    }

    pub fn data_mut(Block(data): &mut Self) -> &mut [bool] {
        data
    }

    pub fn data(Block(data): &Self) -> &[bool] {
        data
    }

    /// Get a subpixel within a block by x and y coordinates
    /// X and Y wrap around block size, which is in units of 2^n
    pub fn index(ker: &dyn Kernel, xy: (i32, i32)) -> usize {
        let block_width = Self::width(ker) as i32;
        let (x, y) = xy;
        let (x, y) = (x % block_width, y % block_width);
        (x + block_width * y) as usize
    }

    pub fn get(&self, ker: &dyn Kernel, xy: (i32, i32)) -> bool {
        let Self(data) = self;
        data[Self::index(ker, xy)]
    }

    pub fn set(&mut self, ker: &dyn Kernel, xy: (i32, i32), val: bool) {
        let Self(data) = self;
        data[Self::index(ker, xy)] = val;
    }
}
