/* CO-RE BTF tracepoints. The small local task_struct is intentionally relocated
 * through BTF; it is not a hardcoded kernel offset layout. No libbpf headers are
 * needed to build this object. The userspace loader uses libbpf public APIs. */
#include <linux/bpf.h>
#include "sched_event.h"
#define SEC(name) __attribute__((section(name), used))
#define __uint(name, value) int (*name)[value]
#define __type(name, value) value *name

struct task_struct {
    int pid;
    int tgid;
} __attribute__((preserve_access_index));

static void *(*bpf_map_lookup_elem)(void *, const void *) = (void *)BPF_FUNC_map_lookup_elem;
static long (*bpf_probe_read_kernel)(void *, __u32, const void *) = (void *)BPF_FUNC_probe_read_kernel;
static __u64 (*bpf_ktime_get_ns)(void) = (void *)BPF_FUNC_ktime_get_ns;
static __u64 (*bpf_get_current_pid_tgid)(void) = (void *)BPF_FUNC_get_current_pid_tgid;
static void *(*bpf_ringbuf_reserve)(void *, __u64, __u64) = (void *)BPF_FUNC_ringbuf_reserve;
static void (*bpf_ringbuf_submit)(void *, __u64) = (void *)BPF_FUNC_ringbuf_submit;

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 64);
    __type(key, __u32);
    __type(value, __u8);
} target_tgids SEC(".maps");
struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 1 << 20);
} events SEC(".maps");
struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, __u64);
} dropped SEC(".maps");

static __attribute__((always_inline)) inline void emit_task(struct task_struct *task,
        __u32 kind, __u64 state, __u32 waker) {
    __u32 tgid = 0, tid = 0;
    bpf_probe_read_kernel(&tgid, sizeof(tgid), __builtin_preserve_access_index(&task->tgid));
    if (!bpf_map_lookup_elem(&target_tgids, &tgid)) return;
    bpf_probe_read_kernel(&tid, sizeof(tid), __builtin_preserve_access_index(&task->pid));
    struct reqscope_sched_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        __u32 key = 0;
        __u64 *counter = bpf_map_lookup_elem(&dropped, &key);
        if (counter) __sync_fetch_and_add(counter, 1);
        return;
    }
    event->timestamp = bpf_ktime_get_ns();
    event->prev_state = state;
    event->tgid = tgid;
    event->tid = tid;
    event->waker_tid = waker;
    event->kind = kind;
    bpf_ringbuf_submit(event, 0);
}

SEC("tp_btf/sched_switch")
int reqscope_sched_switch(unsigned long long *ctx) {
    struct task_struct *previous = (struct task_struct *)ctx[1];
    struct task_struct *next = (struct task_struct *)ctx[2];
    __u64 state = ctx[3];
    emit_task(previous, REQSCOPE_SWITCH_OUT, state, 0);
    emit_task(next, REQSCOPE_SWITCH_IN, 0, 0);
    return 0;
}
SEC("tp_btf/sched_wakeup")
int reqscope_sched_wakeup(unsigned long long *ctx) {
    emit_task((struct task_struct *)ctx[0], REQSCOPE_WAKEUP, 0, (__u32)bpf_get_current_pid_tgid());
    return 0;
}
SEC("tp_btf/sched_wakeup_new")
int reqscope_sched_wakeup_new(unsigned long long *ctx) {
    emit_task((struct task_struct *)ctx[0], REQSCOPE_WAKEUP_NEW, 0, (__u32)bpf_get_current_pid_tgid());
    return 0;
}
char LICENSE[] SEC("license") = "Dual BSD/GPL";
