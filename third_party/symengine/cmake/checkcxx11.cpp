#include <unordered_map>
#include <string>
#include <cstddef>
#include <stdexcept>

template<class T>
class Ptr {
public:
    inline explicit Ptr( T *ptr ) : ptr_(ptr) {
    }
    inline Ptr(const Ptr<T>& ptr) : ptr_(ptr.ptr_) {}
    template<class T2> inline Ptr(const Ptr<T2>& ptr) : ptr_(ptr.get()) {}
    Ptr<T>& operator=(const Ptr<T>& ptr) { ptr_ = ptr.get(); return *this; }
//    Ptr(Ptr&&) = default;
//    Ptr<T>& operator=(Ptr&&) = default;
    inline T* operator->() const { return ptr_; }
    inline T& operator*() const { return *ptr_; }
    inline T* get() const { return ptr_; }
    inline T* getRawPtr() const { return get(); }
    inline const Ptr<T> ptr() const { return *this; }
private:
    T *ptr_;
};

template<typename T> inline
Ptr<T> outArg( T& arg )
{
    return Ptr<T>(&arg);
}

enum ENull { null };

template<class T>
class RCP {
public:
    RCP(ENull null_arg = null) : ptr_(NULL) {}
    explicit RCP(T *p) : ptr_(p) {
        (ptr_->refcount_)++;
    }
    RCP(const RCP<T> &rp) : ptr_(rp.ptr_) {
        if (!is_null()) (ptr_->refcount_)++;
    }
    template<class T2> RCP(const RCP<T2>& r_ptr) : ptr_(r_ptr.get()) {
        if (!is_null()) (ptr_->refcount_)++;
    }
    RCP(RCP<T> &&rp) : ptr_(rp.ptr_) {
        rp.ptr_ = NULL;
    }
    template<class T2> RCP(RCP<T2>&& r_ptr) : ptr_(r_ptr.get()) {
        r_ptr._set_null();
    }
    ~RCP() {
        if (ptr_ != NULL && --(ptr_->refcount_) == 0) delete ptr_;
    }
    T* operator->() const {
        return ptr_;
    }
    T& operator*() const {
        return *ptr_;
    }
    T* get() const { return ptr_; }
    Ptr<T> ptr() const { return Ptr<T>(get()); }
    bool is_null() const { return ptr_ == NULL; }
    template<class T2> bool operator==(const RCP<T2> &p2) {
        return ptr_ == p2.ptr_;
    }
    template<class T2> bool operator!=(const RCP<T2> &p2) {
        return ptr_ != p2.ptr_;
    }
    RCP<T>& operator=(const RCP<T> &r_ptr) {
        T *r_ptr_ptr_ = r_ptr.ptr_;
        if (!r_ptr.is_null()) (r_ptr_ptr_->refcount_)++;
        if (!is_null() && --(ptr_->refcount_) == 0) delete ptr_;
        ptr_ = r_ptr_ptr_;
        return *this;
    }
    RCP<T>& operator=(RCP<T> &&r_ptr) {
        std::swap(ptr_, r_ptr.ptr_);
        return *this;
    }
    void reset() {
        if (!is_null() && --(ptr_->refcount_) == 0) delete ptr_;
        ptr_ = NULL;
    }
    void _set_null() { ptr_ = NULL; }
private:
    T *ptr_;
};

template<class T>
inline RCP<T> rcp(T* p)
{
    return RCP<T>(p);
}

template<class T2, class T1>
inline RCP<T2> rcp_static_cast(const RCP<T1>& p1)
{
    T2 *check = static_cast<T2*>(p1.get());
    return RCP<T2>(check);
}

template<class T2, class T1>
inline RCP<T2> rcp_dynamic_cast(const RCP<T1>& p1)
{
    if (!p1.is_null()) {
        T2 *p = NULL;
        p = dynamic_cast<T2*>(p1.get());
        if (p) {
            return RCP<T2>(p);
        }
    }
    throw std::runtime_error("rcp_dynamic_cast: cannot convert.");
}

template<class T2, class T1>
inline RCP<T2> rcp_const_cast(const RCP<T1>& p1)
{
  T2 *check = const_cast<T2*>(p1.get());
  return RCP<T2>(check);
}


template<class T>
inline bool operator==(const RCP<T> &p, ENull)
{
  return p.get() == NULL;
}


template<typename T>
std::string typeName(const T &t)
{
    return "RCP<>";
}

long double operator "" _mul2(long double x) {
    return 2*x;
}

class A {
public:
    virtual void print();
};

class B : public A {
public:
    virtual void print() override;
};

int main() {
    return 0;
}
