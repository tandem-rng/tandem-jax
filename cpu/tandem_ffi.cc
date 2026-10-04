/* The tandem-c fills as one XLA FFI handler for the CPU, the counterpart of cuda/tandem_ffi.cu
 * with the same operands and attributes.
 *
 * Every fill is random access, so a fill splits into parts at any element boundary, each part
 * a tandem-c fill from its own start position. Large fills run their parts on XLA's intra-op
 * thread pool and return a future, so no pool thread blocks on another.
 *
 * Batched calls (vmap) carry one parameter row per fill and write the fills back to back, row b
 * to out + b n.
 */
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <type_traits>

#include "tandem/tandem.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace {

/* key[4], then position, range and low bound as (low, high) word pairs. */
constexpr int64_t PARAMS = 10;
/* Parts start on whole 1024-bit stream rows for every width, and on whole normal pairs. */
constexpr uint64_t GRAIN = 128;
/* Parts below this many bytes cost more to schedule than to fill. */
constexpr uint64_t MIN_PART_BYTES = 1u << 18;
constexpr uint64_t RANGE_2_32 = 1ull << 32;

enum class Kind { stream, below, normal };

struct Params {
    uint32_t key[4];
    uint64_t pos, range, low;
};

Params load(const uint32_t *p) {
    Params q;
    std::memcpy(q.key, p, sizeof q.key);
    q.pos = p[4] | (uint64_t)p[5] << 32;
    q.range = p[6] | (uint64_t)p[7] << 32;
    q.low = p[8] | (uint64_t)p[9] << 32;
    return q;
}

uint64_t align_pos(uint64_t pos, unsigned w) { return (pos + (w - 1)) & ~(uint64_t)(w - 1); }

/* The generator positioned at element `i` of a fill of `w`-bit draws. */
tandem_rng at(const Params &q, uint32_t K, unsigned w, uint64_t i) {
    return tandem_from_key(q.key, align_pos(q.pos, w) + i * w, K);
}

/* Elements [i0, i1) of `n` bounded draws, `low + [0, range)`, into out[i0..i1) of type O. */
template <class O> void below(const Params &q, uint32_t K, int width, uint64_t i0, uint64_t i1, O *out) {
    using U = std::make_unsigned_t<O>;
    uint64_t m = i1 - i0;
    out += i0;
    if constexpr (sizeof(O) == 8)
        if (width == 64 || (width == 0 && q.range > RANGE_2_32)) {
            tandem_rng r = at(q, K, 64, i0);
            auto *v = reinterpret_cast<uint64_t *>(out);
            tandem_fill_u64_below(&r, v, m, q.range);
            for (uint64_t i = 0; i < m; i++) out[i] = (O)(v[i] + q.low);
            return;
        }
    tandem_rng r = at(q, K, 32, i0);
    if constexpr (sizeof(O) == 4) {
        auto *v = reinterpret_cast<uint32_t *>(out);
        tandem_fill_u32_below(&r, v, m, (uint32_t)q.range);
        for (uint64_t i = 0; i < m; i++) out[i] = (O)(U)(v[i] + (U)q.low);
    } else {
        /* 32-bit draws widen into a 64-bit output through a buffer that stays in L1. */
        uint32_t v[1024];
        for (uint64_t done = 0; done < m;) {
            size_t b = (size_t)std::min<uint64_t>(m - done, 1024);
            if (q.range == RANGE_2_32) tandem_fill_u32(&r, v, b);
            else tandem_fill_u32_below(&r, v, b, (uint32_t)q.range);
            for (size_t i = 0; i < b; i++) out[done + i] = (O)(U)(v[i] + q.low);
            done += b;
        }
    }
}

struct Job {
    const uint32_t *prm;
    void *out;
    ffi::DataType type;
    Kind kind;
    uint32_t K;
    int width;
    uint64_t n;

    /* Elements [i0, i1) of row `row`. */
    void run(uint64_t row, uint64_t i0, uint64_t i1) const {
        using D = ffi::DataType;
        Params q = load(prm + PARAMS * row);
        uint64_t m = i1 - i0, first = n * row + i0;
        switch (kind) {
        case Kind::stream: {
            tandem_rng r = at(q, K, 8 * (unsigned)ffi::ByteWidth(type), i0);
            switch (type) {
            case D::U8: case D::S8: tandem_fill_u8(&r, (uint8_t *)out + first, m); break;
            case D::U16: case D::S16: tandem_fill_u16(&r, (uint16_t *)out + first, m); break;
            case D::F16: tandem_fill_f16_bits(&r, (uint16_t *)out + first, m); break;
            case D::U32: case D::S32: tandem_fill_u32(&r, (uint32_t *)out + first, m); break;
            case D::U64: case D::S64: tandem_fill_u64(&r, (uint64_t *)out + first, m); break;
            case D::F32: tandem_fill_f32(&r, (float *)out + first, m); break;
            default: tandem_fill_f64(&r, (double *)out + first, m); break;
            }
            break;
        }
        case Kind::below:
            switch (type) {
            case D::U32: below(q, K, width, i0, i1, (uint32_t *)out + n * row); break;
            case D::S32: below(q, K, width, i0, i1, (int32_t *)out + n * row); break;
            case D::U64: below(q, K, width, i0, i1, (uint64_t *)out + n * row); break;
            default: below(q, K, width, i0, i1, (int64_t *)out + n * row); break;
            }
            break;
        case Kind::normal: {
            tandem_rng r = at(q, K, 8 * (unsigned)ffi::ByteWidth(type), i0);
            if (type == D::F32) tandem_fill_normal_f32(&r, (float *)out + first, m);
            else tandem_fill_normal_f64(&r, (double *)out + first, m);
            break;
        }
        }
    }

    /* Flat elements [a, b) of all rows, cut at multiples of GRAIN within each row, so that
     * neighbouring parts meet at the same cut. */
    void run_flat(uint64_t a, uint64_t b) const {
        auto cut = [&](uint64_t x, uint64_t row) {
            uint64_t s = n * row;
            return x <= s ? 0 : std::min(n, (x - s + GRAIN - 1) / GRAIN * GRAIN);
        };
        for (uint64_t row = a / n; row * n < b; row++) {
            uint64_t i0 = cut(a, row), i1 = cut(b, row);
            if (i0 < i1) run(row, i0, i1);
        }
    }
};

ffi::Error check(std::string_view kind, ffi::DataType t, int32_t width, Kind &k) {
    using D = ffi::DataType;
    if (kind == "stream") {
        k = Kind::stream;
        switch (t) {
        case D::U8: case D::S8: case D::U16: case D::S16: case D::F16: case D::U32: case D::S32:
        case D::F32: case D::U64: case D::S64: case D::F64: return ffi::Error::Success();
        default: return ffi::Error::InvalidArgument("tandem_fill: stream needs an 8- to 64-bit integer or float output");
        }
    }
    if (kind == "below") {
        k = Kind::below;
        if (t != D::U32 && t != D::S32 && t != D::U64 && t != D::S64)
            return ffi::Error::InvalidArgument("tandem_fill: below needs u32, s32, u64 or s64");
        if (width == 64 && (t == D::U32 || t == D::S32))
            return ffi::Error::InvalidArgument("tandem_fill: 64-bit draws need a 64-bit output");
        return ffi::Error::Success();
    }
    if (kind == "normal") {
        k = Kind::normal;
        if (t != D::F32 && t != D::F64) return ffi::Error::InvalidArgument("tandem_fill: normal needs f32 or f64");
        return ffi::Error::Success();
    }
    return ffi::Error::InvalidArgument("tandem_fill: kind must be stream, below or normal");
}

ffi::Future Fill(ffi::ThreadPool pool, ffi::Buffer<ffi::U32> prm, ffi::Result<ffi::AnyBuffer> out, int64_t n,
                 int64_t chunk, std::string_view kind, int32_t width) {
    ffi::Promise done;
    ffi::Future future(done);
    uint64_t rows = prm.element_count() / PARAMS, m = (uint64_t)n;
    Kind k;
    ffi::Error e = check(kind, out->element_type(), width, k);
    if (prm.element_count() % PARAMS || rows * m != out->element_count())
        e = ffi::Error::InvalidArgument("tandem_fill: parameter rows and output size disagree");
    if (e.failure()) {
        done.SetError(e);
        return future;
    }
    Job job{prm.typed_data(), out->untyped_data(), out->element_type(), k, (uint32_t)chunk, width, m};
    uint64_t total = rows * m;
    uint64_t parts = std::min<uint64_t>((uint64_t)std::max<int64_t>(pool.num_threads(), 1),
                                        total * ffi::ByteWidth(out->element_type()) / MIN_PART_BYTES);
    if (parts <= 1) {
        if (total) job.run_flat(0, total);
        done.SetAvailable();
        return future;
    }
    ffi::CountDownPromise left(std::move(done), (int64_t)parts);
    for (uint64_t p = 0; p < parts; p++)
        pool.Schedule([job, left, a = total * p / parts, b = total * (p + 1) / parts]() mutable {
            job.run_flat(a, b);
            left.CountDown();
        });
    return future;
}

} // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(TandemFill, Fill,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::ThreadPool>()
                                  .Arg<ffi::Buffer<ffi::U32>>()
                                  .Ret<ffi::AnyBuffer>()
                                  .Attr<int64_t>("n")
                                  .Attr<int64_t>("chunk")
                                  .Attr<std::string_view>("kind")
                                  .Attr<int32_t>("width"));
