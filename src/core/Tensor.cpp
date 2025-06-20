#include "../incl/core/Tensor.h"
#include <cassert>
#include <cmath>

// ===== Constructors =====
Tensor::Tensor(int rows, int cols, bool requires_grad)
    : data(Eigen::MatrixXf::Zero(rows, cols)),
      grad(Eigen::MatrixXf::Zero(rows, cols)),
      requires_grad(requires_grad) {}

Tensor::Tensor(const Eigen::MatrixXf& data_, bool requires_grad)
    : data(data_),
      grad(Eigen::MatrixXf::Zero(data_.rows(), data_.cols())),
      requires_grad(requires_grad) {}

// ===== Autograd =====
void Tensor::zero_grad() {
    grad.setZero();
}

void Tensor::backward() {
    // Initialize gradient to ones for the root tensor
    if (grad.size() == 0 || grad.isZero()) {
        grad = Eigen::MatrixXf::Ones(data.rows(), data.cols());
    }

    std::vector<std::shared_ptr<Tensor>> stack = { shared_from_this() };
    while (!stack.empty()) {
        auto cur = stack.back();
        stack.pop_back();
        if (cur->grad_fn) {
            cur->grad_fn->backward(cur);
            for (auto& parent : cur->grad_fn->parents) {
                if (parent->requires_grad) {
                    stack.push_back(parent);
                }
            }
        }
    }
}

// ===== Core Ops =====
std::shared_ptr<Tensor> Tensor::operator+(const std::shared_ptr<Tensor>& other) {
    assert(data.rows() == other->data.rows() && data.cols() == other->data.cols());
    bool req_grad = requires_grad || other->requires_grad;
    auto out = std::make_shared<Tensor>(data + other->data, req_grad);

    if (req_grad) {
        out->grad_fn = std::make_shared<GradFn>();
        out->grad_fn->op_type = "add";
        out->grad_fn->parents = {shared_from_this(), other};
        out->grad_fn->backward = [a=shared_from_this(), b=other](const std::shared_ptr<Tensor>& self) {
            if (a->requires_grad) a->grad += self->grad;
            if (b->requires_grad) b->grad += self->grad;
        };
    }

    return out;
}

std::shared_ptr<Tensor> Tensor::operator-(const std::shared_ptr<Tensor>& other) {
    assert(data.rows() == other->data.rows() && data.cols() == other->data.cols());
    auto out = std::make_shared<Tensor>(data - other->data, requires_grad || other->requires_grad);

    if (out->requires_grad) {
        out->grad_fn = std::make_shared<GradFn>();
        out->grad_fn->op_type = "sub";
        out->grad_fn->parents = {shared_from_this(), other};
        out->grad_fn->backward = [a=shared_from_this(), b=other](const std::shared_ptr<Tensor>& self) {
            if (a->requires_grad) a->grad += self->grad;
            if (b->requires_grad) b->grad -= self->grad;
        };
    }

    return out;
}

std::shared_ptr<Tensor> Tensor::operator/(float scalar) {
    auto out = std::make_shared<Tensor>(data / scalar, requires_grad);

    if (requires_grad) {
        out->grad_fn = std::make_shared<GradFn>();
        out->grad_fn->op_type = "divide";
        out->grad_fn->parents = {shared_from_this()};
        out->grad_fn->backward = [scalar, a=shared_from_this()](const std::shared_ptr<Tensor>& self) {
            if (a->requires_grad) a->grad += self->grad / scalar;
        };
    }

    return out;
}

std::shared_ptr<Tensor> Tensor::operator*(float scalar) {
    auto out = std::make_shared<Tensor>(data * scalar, requires_grad);

    if (requires_grad) {
        out->grad_fn = std::make_shared<GradFn>();
        out->grad_fn->op_type = "scalar multiply";
        out->grad_fn->parents = {shared_from_this()};
        out->grad_fn->backward = [scalar, a=shared_from_this()](const std::shared_ptr<Tensor>& self) {
            if (a->requires_grad) a->grad += self->grad * scalar;
        };
    }

    return out;
}


std::shared_ptr<Tensor> Tensor::matmul(const std::shared_ptr<Tensor>& other) {
    assert(data.cols() == other->data.rows());
    bool req_grad = requires_grad || other->requires_grad;
    auto out = std::make_shared<Tensor>(data * other->data, req_grad);

    if (req_grad) {
        out->grad_fn = std::make_shared<GradFn>();
        out->grad_fn->op_type = "matmul";
        out->grad_fn->parents = {shared_from_this(), other};
        out->grad_fn->backward = [a=shared_from_this(), b=other](const std::shared_ptr<Tensor>& self) {
            if (a->requires_grad) a->grad += self->grad * b->data.transpose();
            if (b->requires_grad) b->grad += a->data.transpose() * self->grad;
        };
    }

    return out;
}

std::shared_ptr<Tensor> Tensor::relu() {
    Eigen::MatrixXf result = data.cwiseMax(0.0f);
    bool req_grad = requires_grad;
    auto out = std::make_shared<Tensor>(result, req_grad);

    if (req_grad) {
        out->grad_fn = std::make_shared<GradFn>();
        out->grad_fn->op_type = "relu";
        out->grad_fn->parents = {shared_from_this()};
        out->grad_fn->backward = [a=shared_from_this()](const std::shared_ptr<Tensor>& self) {
            Eigen::MatrixXf mask = (a->data.array() > 0.0f).cast<float>();
            if (a->requires_grad) a->grad += self->grad.cwiseProduct(mask);
        };
    }

    return out;
}

std::shared_ptr<Tensor> Tensor::sin() {
    Eigen::MatrixXf result = data.array().sin();
    bool req_grad = requires_grad;
    auto out = std::make_shared<Tensor>(result, req_grad);

    if (req_grad) {
        out->grad_fn = std::make_shared<GradFn>();
        out->grad_fn->op_type = "sin";
        out->grad_fn->parents = {shared_from_this()};
        out->grad_fn->backward = [a=shared_from_this()](const std::shared_ptr<Tensor>& self) {
            if (a->requires_grad) {
                Eigen::MatrixXf deriv = a->data.array().cos();
                a->grad += self->grad.cwiseProduct(deriv);
            }
        };
    }

    return out;
}

std::shared_ptr<Tensor> Tensor::cos() {
    Eigen::MatrixXf result = data.array().cos();
    bool req_grad = requires_grad;
    auto out = std::make_shared<Tensor>(result, req_grad);

    if (req_grad) {
        out->grad_fn = std::make_shared<GradFn>();
        out->grad_fn->op_type = "cos";
        out->grad_fn->parents = {shared_from_this()};
        out->grad_fn->backward = [a=shared_from_this()](const std::shared_ptr<Tensor>& self) {
            if (a->requires_grad) {
                Eigen::MatrixXf deriv = -a->data.array().sin();
                a->grad += self->grad.cwiseProduct(deriv);
            }
        };
    }

    return out;
}


std::vector<std::shared_ptr<Tensor>> Tensor::split_heads(int num_heads) {
    int B_T = data.rows();
    int d_model = data.cols();
    assert(d_model % num_heads == 0);
    int d_k = d_model / num_heads;

    std::vector<std::shared_ptr<Tensor>> heads;
    for (int h = 0; h < num_heads; ++h) {
        Eigen::MatrixXf slice = data.middleCols(h * d_k, d_k);
        heads.push_back(std::make_shared<Tensor>(slice, requires_grad));
    }
    return heads;
}

std::shared_ptr<Tensor> Tensor::concat_heads(const std::vector<std::shared_ptr<Tensor>>& heads) {
    int rows = heads[0]->data.rows();
    int total_cols = 0;
    for (const auto& h : heads) total_cols += h->data.cols();

    Eigen::MatrixXf result(rows, total_cols);
    int col_offset = 0;
    for (const auto& h : heads) {
        result.middleCols(col_offset, h->data.cols()) = h->data;
        col_offset += h->data.cols();
    }
    return std::make_shared<Tensor>(result);
} 


std::shared_ptr<Tensor> Tensor::transpose() {
    auto out = std::make_shared<Tensor>(data.transpose(), requires_grad);

    // May not need backwards/gradient tracking
    if (requires_grad) {
        out->grad_fn = std::make_shared<GradFn>();
        out->grad_fn->op_type = "transpose";
        out->grad_fn->parents = {shared_from_this()};
        out->grad_fn->backward = [a=shared_from_this()](const std::shared_ptr<Tensor>& self) {
            if (a->requires_grad) a->grad += self->grad.transpose();
        };
    }

    return out;
}

std::shared_ptr<Tensor> Tensor::reshape(int new_rows, int new_cols) {
    assert(new_rows * new_cols == data.size());
    Eigen::MatrixXf reshaped = Eigen::Map<const Eigen::MatrixXf>(data.data(), new_rows, new_cols);
    return std::make_shared<Tensor>(reshaped, requires_grad);
}



// ===== Utilities =====
std::pair<int, int> Tensor::shape() const {
    return { data.rows(), data.cols() };
}

void Tensor::print(const std::string& label) const {
    if (!label.empty()) std::cout << label << std::endl;
    std::cout << data << std::endl;
}

std::shared_ptr<Tensor> Tensor::clone() const {
    auto cloned = std::make_shared<Tensor>(data, requires_grad);
    cloned->grad = grad;
    cloned->grad_fn = grad_fn;
    return cloned;
}

std::shared_ptr<Tensor> Tensor::detach() const {
    return std::make_shared<Tensor>(data, false);
}
