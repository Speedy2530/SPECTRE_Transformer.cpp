#include "../incl/Embedding.h"
#include <random>
#include <stdexcept>

Embedding::Embedding(int vocab_size, int d_model)
    : embedding_table(vocab_size, d_model, true) {

    // Xavier Initialization (normal dist with stddev = 1/sqrt(d_model))
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0.0, 1.0 / std::sqrt(d_model));

    for (int i = 0; i < vocab_size; ++i) {
        for (int j = 0; j < d_model; ++j) {
            embedding_table.data(i, j) = distribution(generator);
        }
    }
}

Tensor Embedding::forward(const std::vector<int>& token_ids) {
    int seq_len = token_ids.size();
    int d_model = embedding_table.data.cols();

    Tensor output(seq_len, d_model, true);  // output requires grad for backprop

    for (int i = 0; i < seq_len; ++i) {
        int id = token_ids[i];
        if (id < 0 || id >= embedding_table.data.rows()) { throw std::out_of_range("Token ID out of bounds."); }
        output.data.row(i) = embedding_table.data.row(id);
    }

    return output;
}

void Embedding::zero_grad() {
    embedding_table.zero_grad();
}

Tensor& Embedding::weights() {
    return embedding_table;
}
