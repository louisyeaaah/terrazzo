"""Small block pool fixture with one known reference-count leak."""


class BlockPool:
    def __init__(self, blocks):
        self.free_blocks = list(blocks)
        self.refcounts = {block: 0 for block in blocks}

    def acquire(self, block):
        if block not in self.free_blocks:
            raise RuntimeError(f"block {block} is not free")
        self.free_blocks.remove(block)
        self.refcounts[block] += 1
        return block

    def release(self, block):
        if self.refcounts[block] <= 0:
            raise RuntimeError(f"block {block} is not owned")
        self.refcounts[block] -= 1
        # Known defect: when the count reaches zero this condition is false, so
        # the block is never returned to free_blocks.
        if self.refcounts[block] == 1:
            self.free_blocks.append(block)
