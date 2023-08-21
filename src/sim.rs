pub type NodeIdx = usize;

/// Block data, whose size is known by the Kernel
pub type Block = Box<[NodeIdx]>;

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

