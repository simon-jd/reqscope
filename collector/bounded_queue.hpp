#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

namespace reqscope::collector {
// Bounded MPSC sequence queue. A producer reserves a cell only when it is free;
// publication uses release/acquire, so the consumer never sees a partial copy.
// All storage is allocated before capture. try_push never waits for another
// producer to complete its copy and never acquires an application mutex.
template<class T> class BoundedQueue {
    struct Cell { std::atomic<size_t> sequence{0}; T value{}; };
    const size_t capacity_;
    std::unique_ptr<Cell[]> cells_;
    alignas(64) std::atomic<size_t> write_{0};
    alignas(64) size_t read_ = 0;
public:
    explicit BoundedQueue(size_t capacity) : capacity_(capacity), cells_(new Cell[capacity]) {
        for (size_t i = 0; i < capacity; ++i) cells_[i].sequence.store(i);
    }
    template<class F> bool try_push(F fill) noexcept {
        size_t pos = write_.load(std::memory_order_relaxed);
        // A contention bound prevents unbounded callback retry under overload.
        for (unsigned attempt = 0; attempt < 32; ++attempt) {
            Cell& cell = cells_[pos % capacity_];
            const intptr_t diff = static_cast<intptr_t>(cell.sequence.load(std::memory_order_acquire))
                                - static_cast<intptr_t>(pos);
            if (diff == 0) {
                if (write_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
                    fill(cell.value);
                    cell.sequence.store(pos + 1, std::memory_order_release);
                    return true;
                }
            } else if (diff < 0) return false;
            else pos = write_.load(std::memory_order_relaxed);
        }
        return false;
    }
    template<class F> bool try_pop(F consume) {
        Cell& cell = cells_[read_ % capacity_];
        if (cell.sequence.load(std::memory_order_acquire) != read_ + 1) return false;
        consume(cell.value);
        cell.sequence.store(read_ + capacity_, std::memory_order_release);
        ++read_;
        return true;
    }
    bool empty() const noexcept {
        return read_ == write_.load(std::memory_order_acquire);
    }
};
}
