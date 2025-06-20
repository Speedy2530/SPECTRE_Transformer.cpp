#include "../incl/PosEncode.h"
#include <cmath>


// Inspired by "Attention is all you need", fixed positional encoding with sin/cos
PositionalEncoding::PositionalEncoding(int max_len, int d_model)
    : pe(max_len, d_model, false) {
    for (int pos = 0; pos < max_len; ++pos) {
        for (int i = 0; i < d_model; ++i) {
            float angle = pos / std::pow(10000.0, 2 * (i / 2) / static_cast<float>(d_model));
            if (i % 2 == 0) {
                pe.data(pos, i) = std::sin(angle);
            } else {
                pe.data(pos, i) = std::cos(angle);
            }
        }
    }
}

Tensor PositionalEncoding::get_encoding(int seq_len) const {
    Tensor out(seq_len, pe.data.cols(), false);
    out.data = pe.data.topRows(seq_len);
    return out;
}
