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

std::shared_ptr<Tensor> Tensor::operator*(float scalar) {
    bool req_grad = requires_grad;
    auto out = std::make_shared<Tensor>(data * scalar, req_grad);

    if (req_grad) {
        out->grad_fn = std::make_shared<GradFn>();
        out->grad_fn->op_type = "scale";
        out->grad_fn->parents = {shared_from_this()};
        out->grad_fn->backward = [scalar, a=shared_from_this()](const std::shared_ptr<Tensor>& self) {
            if (a->requires_grad) a->grad += self->grad * scalar;
        };
    }

    return out;
}

std::shared_ptr<Tensor> Tensor::dot(const std::shared_ptr<Tensor>& other) {
    assert(data.cols() == other->data.rows());
    bool req_grad = requires_grad || other->requires_grad;
    auto out = std::make_shared<Tensor>(data * other->data, req_grad);

    if (req_grad) {
        out->grad_fn = std::make_shared<GradFn>();
        out->grad_fn->op_type = "dot";
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
