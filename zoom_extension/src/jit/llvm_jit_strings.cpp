// This is copy-pasted (with modification) from the following llvm file:
// - https://github.com/llvm/llvm-project/blob/main/libcxx/include/complex
//
// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <string>
#include "llvm_jit_strings.h"


namespace at::zoom {

// copy-pasted from some llvm files:
// - https://github.com/llvm/llvm-project/blob/main/libcxx/include/type_traits
// - https://github.com/llvm/llvm-project/blob/main/clang/test/Headers/Inputs/include/type_traits

// hiprtc already includes some traits, so this removes duplicate definitions of
// integral_constant, is_same, is_integral, enable_if, is_floating_point, is_arithmetic.
// Copied from aten/src/ATen/cuda/llvm_basic.cpp, then modified as above.
const std::string traits = R"ESCAPE(
namespace std {

template <class _Tp>
_Tp&& __declval(int);
template <class _Tp>
_Tp __declval(long);
template <class _Tp>
decltype(__declval<_Tp>(0)) declval() noexcept;

template <class _Tp> struct remove_const            {typedef _Tp type;};
template <class _Tp> struct remove_const<const _Tp> {typedef _Tp type;};
template <class _Tp> using remove_const_t = typename remove_const<_Tp>::type;

template <class _Tp> struct remove_volatile               {typedef _Tp type;};
template <class _Tp> struct remove_volatile<volatile _Tp> {typedef _Tp type;};
template <class _Tp> using remove_volatile_t = typename remove_volatile<_Tp>::type;

template <class _Tp> struct remove_cv
{typedef typename remove_volatile<typename remove_const<_Tp>::type>::type type;};
template <class _Tp> using remove_cv_t = typename remove_cv<_Tp>::type;

template <class _Tp> struct __libcpp_is_floating_point              : public false_type {};
template <>          struct __libcpp_is_floating_point<float>       : public true_type {};
template <>          struct __libcpp_is_floating_point<double>      : public true_type {};
template <>          struct __libcpp_is_floating_point<long double> : public true_type {};

template <class _Tp>
inline constexpr bool is_arithmetic_v = is_arithmetic<_Tp>::value;

template <class _Tp>
struct __numeric_type
{
   static void __test(...);
   static float __test(float);
   static double __test(char);
   static double __test(int);
   static double __test(unsigned);
   static double __test(long);
   static double __test(unsigned long);
   static double __test(long long);
   static double __test(unsigned long long);
   static double __test(double);
   static long double __test(long double);

   typedef decltype(__test(declval<_Tp>())) type;
   static const bool value = !is_same<type, void>::value;
};

template <>
struct __numeric_type<void>
{
   static const bool value = true;
};

// __promote

template <class _A1, class _A2 = void, class _A3 = void,
          bool = __numeric_type<_A1>::value &&
                 __numeric_type<_A2>::value &&
                 __numeric_type<_A3>::value>
class __promote_imp
{
public:
    static const bool value = false;
};

template <class _A1, class _A2, class _A3>
class __promote_imp<_A1, _A2, _A3, true>
{
private:
    typedef typename __promote_imp<_A1>::type __type1;
    typedef typename __promote_imp<_A2>::type __type2;
    typedef typename __promote_imp<_A3>::type __type3;
public:
    typedef decltype(__type1() + __type2() + __type3()) type;
    static const bool value = true;
};

template <class _A1, class _A2>
class __promote_imp<_A1, _A2, void, true>
{
private:
    typedef typename __promote_imp<_A1>::type __type1;
    typedef typename __promote_imp<_A2>::type __type2;
public:
    typedef decltype(__type1() + __type2()) type;
    static const bool value = true;
};

template <class _A1>
class __promote_imp<_A1, void, void, true>
{
public:
    typedef typename __numeric_type<_A1>::type type;
    static const bool value = true;
};

template <class _A1, class _A2 = void, class _A3 = void>
class __promote : public __promote_imp<_A1, _A2, _A3> {};

} // namespace std
)ESCAPE";

const std::string &get_traits_string() {
    return traits;
}

// This is copy-pasted from the following llvm file:
// - https://github.com/llvm/llvm-project/blob/main/libcxx/include/cmath
const std::string cmath = R"ESCAPE(

namespace std {

using ::signbit;
using ::isfinite;
using ::isinf;
using ::isnan;

using ::abs;

using ::acos;
using ::acosf;
using ::asin;
using ::asinf;
using ::atan;
using ::atanf;
using ::atan2;
using ::atan2f;
using ::ceil;
using ::ceilf;
using ::cos;
using ::cosf;
using ::cosh;
using ::coshf;

using ::exp;
using ::expf;

using ::fabs;
using ::fabsf;
using ::floor;
using ::floorf;

using ::fmod;
using ::fmodf;

using ::frexp;
using ::frexpf;
using ::ldexp;
using ::ldexpf;

using ::log;
using ::logf;

using ::log10;
using ::log10f;
using ::modf;
using ::modff;

using ::pow;
using ::powf;

using ::sin;
using ::sinf;
using ::sinh;
using ::sinhf;

using ::sqrt;
using ::sqrtf;
using ::tan;
using ::tanf;

using ::tanh;
using ::tanhf;

using ::acosh;
using ::acoshf;
using ::asinh;
using ::asinhf;
using ::atanh;
using ::atanhf;
using ::cbrt;
using ::cbrtf;

using ::copysign;
using ::copysignf;

using ::erf;
using ::erff;
using ::erfc;
using ::erfcf;
using ::exp2;
using ::exp2f;
using ::expm1;
using ::expm1f;
using ::fdim;
using ::fdimf;
using ::fmaf;
using ::fma;
using ::fmax;
using ::fmaxf;
using ::fmin;
using ::fminf;
using ::hypot;
using ::hypotf;
using ::ilogb;
using ::ilogbf;
using ::lgamma;
using ::lgammaf;
using ::llrint;
using ::llrintf;
using ::llround;
using ::llroundf;
using ::log1p;
using ::log1pf;
using ::log2;
using ::log2f;
using ::logb;
using ::logbf;
using ::lrint;
using ::lrintf;
using ::lround;
using ::lroundf;

using ::nan;
using ::nanf;

using ::nearbyint;
using ::nearbyintf;
using ::nextafter;
using ::nextafterf;
using ::remainder;
using ::remainderf;
using ::remquo;
using ::remquof;
using ::rint;
using ::rintf;
using ::round;
using ::roundf;
using ::scalbln;
using ::scalblnf;
using ::scalbn;
using ::scalbnf;
using ::tgamma;
using ::tgammaf;
using ::trunc;
using ::truncf;

} // namespace std

)ESCAPE";

const std::string &get_cmath_string() {
    return cmath;
}


const std::string complex_body = R"ESCAPE(

namespace std {

template<class _Tp> class complex;

template<class _Tp> complex<_Tp> operator*(const complex<_Tp>& __z, const complex<_Tp>& __w);
template<class _Tp> complex<_Tp> operator/(const complex<_Tp>& __x, const complex<_Tp>& __y);

template<class _Tp>
class complex
{
public:
    typedef _Tp value_type;
private:
    value_type __re_;
    value_type __im_;
public:
    constexpr
    complex(const value_type& __re = value_type(), const value_type& __im = value_type())
        : __re_(__re), __im_(__im) {}
    template<class _Xp> constexpr
    complex(const complex<_Xp>& __c)
        : __re_(__c.real()), __im_(__c.imag()) {}

    constexpr value_type real() const {return __re_;}
    constexpr value_type imag() const {return __im_;}

    void real(value_type __re) {__re_ = __re;}
    void imag(value_type __im) {__im_ = __im;}

    constexpr operator bool() const {
        return real() || imag();
    }

    complex& operator= (const value_type& __re)
        {__re_ = __re; __im_ = value_type(); return *this;}
    complex& operator+=(const value_type& __re) {__re_ += __re; return *this;}
    complex& operator-=(const value_type& __re) {__re_ -= __re; return *this;}
    complex& operator*=(const value_type& __re) {__re_ *= __re; __im_ *= __re; return *this;}
    complex& operator/=(const value_type& __re) {__re_ /= __re; __im_ /= __re; return *this;}

    template<class _Xp> complex& operator= (const complex<_Xp>& __c)
        {
            __re_ = __c.real();
            __im_ = __c.imag();
            return *this;
        }
    template<class _Xp> complex& operator+=(const complex<_Xp>& __c)
        {
            __re_ += __c.real();
            __im_ += __c.imag();
            return *this;
        }
    template<class _Xp> complex& operator-=(const complex<_Xp>& __c)
        {
            __re_ -= __c.real();
            __im_ -= __c.imag();
            return *this;
        }
    template<class _Xp> complex& operator*=(const complex<_Xp>& __c)
        {
            *this = *this * complex(__c.real(), __c.imag());
            return *this;
        }
    template<class _Xp> complex& operator/=(const complex<_Xp>& __c)
        {
            *this = *this / complex(__c.real(), __c.imag());
            return *this;
        }
};

template<> class complex<double>;

template<>
class complex<float>
{
    float __re_;
    float __im_;
public:
    typedef float value_type;

    constexpr complex(float __re = 0.0f, float __im = 0.0f)
        : __re_(__re), __im_(__im) {}

    explicit constexpr complex(const complex<double>& __c);

    constexpr float real() const {return __re_;}
    constexpr float imag() const {return __im_;}

    void real(value_type __re) {__re_ = __re;}
    void imag(value_type __im) {__im_ = __im;}

    constexpr operator bool() const {
        return real() || imag();
    }

    complex& operator= (float __re)
        {__re_ = __re; __im_ = value_type(); return *this;}
    complex& operator+=(float __re) {__re_ += __re; return *this;}
    complex& operator-=(float __re) {__re_ -= __re; return *this;}
    complex& operator*=(float __re) {__re_ *= __re; __im_ *= __re; return *this;}
    complex& operator/=(float __re) {__re_ /= __re; __im_ /= __re; return *this;}

    template<class _Xp> complex& operator= (const complex<_Xp>& __c)
        {
            __re_ = __c.real();
            __im_ = __c.imag();
            return *this;
        }
    template<class _Xp> complex& operator+=(const complex<_Xp>& __c)
        {
            __re_ += __c.real();
            __im_ += __c.imag();
            return *this;
        }
    template<class _Xp> complex& operator-=(const complex<_Xp>& __c)
        {
            __re_ -= __c.real();
            __im_ -= __c.imag();
            return *this;
        }
    template<class _Xp> complex& operator*=(const complex<_Xp>& __c)
        {
            *this = *this * complex(__c.real(), __c.imag());
            return *this;
        }
    template<class _Xp> complex& operator/=(const complex<_Xp>& __c)
        {
            *this = *this / complex(__c.real(), __c.imag());
            return *this;
        }
};

template<>
class complex<double>
{
    double __re_;
    double __im_;
public:
    typedef double value_type;

    constexpr complex(double __re = 0.0, double __im = 0.0)
        : __re_(__re), __im_(__im) {}

    constexpr complex(const complex<float>& __c);

    constexpr double real() const {return __re_;}
    constexpr double imag() const {return __im_;}

    void real(value_type __re) {__re_ = __re;}
    void imag(value_type __im) {__im_ = __im;}

    constexpr operator bool() const {
        return real() || imag();
    }

    complex& operator= (double __re)
        {__re_ = __re; __im_ = value_type(); return *this;}
    complex& operator+=(double __re) {__re_ += __re; return *this;}
    complex& operator-=(double __re) {__re_ -= __re; return *this;}
    complex& operator*=(double __re) {__re_ *= __re; __im_ *= __re; return *this;}
    complex& operator/=(double __re) {__re_ /= __re; __im_ /= __re; return *this;}

    template<class _Xp> complex& operator= (const complex<_Xp>& __c)
        {
            __re_ = __c.real();
            __im_ = __c.imag();
            return *this;
        }
    template<class _Xp> complex& operator+=(const complex<_Xp>& __c)
        {
            __re_ += __c.real();
            __im_ += __c.imag();
            return *this;
        }
    template<class _Xp> complex& operator-=(const complex<_Xp>& __c)
        {
            __re_ -= __c.real();
            __im_ -= __c.imag();
            return *this;
        }
    template<class _Xp> complex& operator*=(const complex<_Xp>& __c)
        {
            *this = *this * complex(__c.real(), __c.imag());
            return *this;
        }
    template<class _Xp> complex& operator/=(const complex<_Xp>& __c)
        {
            *this = *this / complex(__c.real(), __c.imag());
            return *this;
        }
};

inline
constexpr
complex<float>::complex(const complex<double>& __c)
    : __re_(__c.real()), __im_(__c.imag()) {}

inline
constexpr
complex<double>::complex(const complex<float>& __c)
    : __re_(__c.real()), __im_(__c.imag()) {}


// 26.3.6 operators:

template<class _Tp>
inline
complex<_Tp>
operator+(const complex<_Tp>& __x, const complex<_Tp>& __y)
{
    complex<_Tp> __t(__x);
    __t += __y;
    return __t;
}

template<class _Tp>
inline
complex<_Tp>
operator+(const complex<_Tp>& __x, const _Tp& __y)
{
    complex<_Tp> __t(__x);
    __t += __y;
    return __t;
}

template<class _Tp>
inline
complex<_Tp>
operator+(const _Tp& __x, const complex<_Tp>& __y)
{
    complex<_Tp> __t(__y);
    __t += __x;
    return __t;
}

template<class _Tp>
inline
complex<_Tp>
operator-(const complex<_Tp>& __x, const complex<_Tp>& __y)
{
    complex<_Tp> __t(__x);
    __t -= __y;
    return __t;
}

template<class _Tp>
inline
complex<_Tp>
operator-(const complex<_Tp>& __x, const _Tp& __y)
{
    complex<_Tp> __t(__x);
    __t -= __y;
    return __t;
}

template<class _Tp>
inline
complex<_Tp>
operator-(const _Tp& __x, const complex<_Tp>& __y)
{
    complex<_Tp> __t(-__y);
    __t += __x;
    return __t;
}

template<class _Tp>
complex<_Tp>
operator*(const complex<_Tp>& __z, const complex<_Tp>& __w)
{
    _Tp __a = __z.real();
    _Tp __b = __z.imag();
    _Tp __c = __w.real();
    _Tp __d = __w.imag();
    _Tp __ac = __a * __c;
    _Tp __bd = __b * __d;
    _Tp __ad = __a * __d;
    _Tp __bc = __b * __c;
    _Tp __x = __ac - __bd;
    _Tp __y = __ad + __bc;
    if (isnan(__x) && isnan(__y))
    {
        bool __recalc = false;
        if (isinf(__a) || isinf(__b))
        {
            __a = copysign(isinf(__a) ? _Tp(1) : _Tp(0), __a);
            __b = copysign(isinf(__b) ? _Tp(1) : _Tp(0), __b);
            if (isnan(__c))
                __c = copysign(_Tp(0), __c);
            if (isnan(__d))
                __d = copysign(_Tp(0), __d);
            __recalc = true;
        }
        if (isinf(__c) || isinf(__d))
        {
            __c = copysign(isinf(__c) ? _Tp(1) : _Tp(0), __c);
            __d = copysign(isinf(__d) ? _Tp(1) : _Tp(0), __d);
            if (isnan(__a))
                __a = copysign(_Tp(0), __a);
            if (isnan(__b))
                __b = copysign(_Tp(0), __b);
            __recalc = true;
        }
        if (!__recalc && (isinf(__ac) || isinf(__bd) ||
                          isinf(__ad) || isinf(__bc)))
        {
            if (isnan(__a))
                __a = copysign(_Tp(0), __a);
            if (isnan(__b))
                __b = copysign(_Tp(0), __b);
            if (isnan(__c))
                __c = copysign(_Tp(0), __c);
            if (isnan(__d))
                __d = copysign(_Tp(0), __d);
            __recalc = true;
        }
        if (__recalc)
        {
            __x = _Tp(INFINITY) * (__a * __c - __b * __d);
            __y = _Tp(INFINITY) * (__a * __d + __b * __c);
        }
    }
    return complex<_Tp>(__x, __y);
}

template<class _Tp>
inline
complex<_Tp>
operator*(const complex<_Tp>& __x, const _Tp& __y)
{
    complex<_Tp> __t(__x);
    __t *= __y;
    return __t;
}

template<class _Tp>
inline
complex<_Tp>
operator*(const _Tp& __x, const complex<_Tp>& __y)
{
    complex<_Tp> __t(__y);
    __t *= __x;
    return __t;
}

template<class _Tp>
complex<_Tp>
operator/(const complex<_Tp>& __z, const complex<_Tp>& __w)
{
    int __ilogbw = 0;
    _Tp __a = __z.real();
    _Tp __b = __z.imag();
    _Tp __c = __w.real();
    _Tp __d = __w.imag();
    _Tp __logbw = logb(fmax(fabs(__c), fabs(__d)));
    if (isfinite(__logbw))
    {
        __ilogbw = static_cast<int>(__logbw);
        __c = scalbn(__c, -__ilogbw);
        __d = scalbn(__d, -__ilogbw);
    }
    _Tp __denom = __c * __c + __d * __d;
    _Tp __x = scalbn((__a * __c + __b * __d) / __denom, -__ilogbw);
    _Tp __y = scalbn((__b * __c - __a * __d) / __denom, -__ilogbw);
    if (isnan(__x) && isnan(__y))
    {
        if ((__denom == _Tp(0)) && (!isnan(__a) || !isnan(__b)))
        {
            __x = copysign(_Tp(INFINITY), __c) * __a;
            __y = copysign(_Tp(INFINITY), __c) * __b;
        }
        else if ((isinf(__a) || isinf(__b)) && isfinite(__c) && isfinite(__d))
        {
            __a = copysign(isinf(__a) ? _Tp(1) : _Tp(0), __a);
            __b = copysign(isinf(__b) ? _Tp(1) : _Tp(0), __b);
            __x = _Tp(INFINITY) * (__a * __c + __b * __d);
            __y = _Tp(INFINITY) * (__b * __c - __a * __d);
        }
        else if (isinf(__logbw) && __logbw > _Tp(0) && isfinite(__a) && isfinite(__b))
        {
            __c = copysign(isinf(__c) ? _Tp(1) : _Tp(0), __c);
            __d = copysign(isinf(__d) ? _Tp(1) : _Tp(0), __d);
            __x = _Tp(0) * (__a * __c + __b * __d);
            __y = _Tp(0) * (__b * __c - __a * __d);
        }
    }
    return complex<_Tp>(__x, __y);
}

template<class _Tp>
inline
complex<_Tp>
operator/(const complex<_Tp>& __x, const _Tp& __y)
{
    return complex<_Tp>(__x.real() / __y, __x.imag() / __y);
}

template<class _Tp>
inline
complex<_Tp>
operator/(const _Tp& __x, const complex<_Tp>& __y)
{
    complex<_Tp> __t(__x);
    __t /= __y;
    return __t;
}

template<class _Tp>
inline
complex<_Tp>
operator+(const complex<_Tp>& __x)
{
    return __x;
}

template<class _Tp>
inline
complex<_Tp>
operator-(const complex<_Tp>& __x)
{
    return complex<_Tp>(-__x.real(), -__x.imag());
}

template<class _Tp>
inline constexpr
bool
operator==(const complex<_Tp>& __x, const complex<_Tp>& __y)
{
    return __x.real() == __y.real() && __x.imag() == __y.imag();
}

template<class _Tp>
inline constexpr
bool
operator==(const complex<_Tp>& __x, const _Tp& __y)
{
    return __x.real() == __y && __x.imag() == 0;
}

template<class _Tp>
inline constexpr
bool
operator==(const _Tp& __x, const complex<_Tp>& __y)
{
    return __x == __y.real() && 0 == __y.imag();
}

template<class _Tp>
inline constexpr
bool
operator!=(const complex<_Tp>& __x, const complex<_Tp>& __y)
{
    return !(__x == __y);
}

template<class _Tp>
inline constexpr
bool
operator!=(const complex<_Tp>& __x, const _Tp& __y)
{
    return !(__x == __y);
}

template<class _Tp>
inline constexpr
bool
operator!=(const _Tp& __x, const complex<_Tp>& __y)
{
    return !(__x == __y);
}

template<class _Tp>
inline constexpr
bool
operator&&(const complex<_Tp>& __x, const complex<_Tp>& __y)
{
    return bool(__x) && bool(__y);
}

template<class _Tp>
inline constexpr
bool
isnan(const complex<_Tp>& __x)
{
    return isnan(__x.real()) || isnan(__x.imag());
}

template<class _Tp>
inline constexpr
bool
operator||(const complex<_Tp>& __x, const complex<_Tp>& __y)
{
    return bool(__x) || bool(__y);
}

// 26.3.7 values:

template <class _Tp, bool = is_integral<_Tp>::value,
                     bool = is_floating_point<_Tp>::value
                     >
struct __libcpp_complex_overload_traits {};

// Integral Types
template <class _Tp>
struct __libcpp_complex_overload_traits<_Tp, true, false>
{
    typedef double _ValueType;
    typedef complex<double> _ComplexType;
};

// Floating point types
template <class _Tp>
struct __libcpp_complex_overload_traits<_Tp, false, true>
{
    typedef _Tp _ValueType;
    typedef complex<_Tp> _ComplexType;
};

// real

template<class _Tp>
inline constexpr
_Tp
real(const complex<_Tp>& __c)
{
    return __c.real();
}

template <class _Tp>
inline constexpr
typename __libcpp_complex_overload_traits<_Tp>::_ValueType
real(_Tp __re)
{
    return __re;
}

// imag

template<class _Tp>
inline constexpr
_Tp
imag(const complex<_Tp>& __c)
{
    return __c.imag();
}

template <class _Tp>
inline constexpr
typename __libcpp_complex_overload_traits<_Tp>::_ValueType
imag(_Tp)
{
    return 0;
}

// abs

template<class _Tp>
inline
_Tp
abs(const complex<_Tp>& __c)
{
    return hypot(__c.real(), __c.imag());
}

// arg

template<class _Tp>
inline
_Tp
arg(const complex<_Tp>& __c)
{
    return atan2(__c.imag(), __c.real());
}

template<class _Tp>
inline
typename enable_if
<
    is_integral<_Tp>::value || is_same<_Tp, double>::value,
    double
>::type
arg(_Tp __re)
{
    return atan2(0., __re);
}

template <class _Tp>
inline
typename enable_if<
    is_same<_Tp, float>::value,
    float
>::type
arg(_Tp __re)
{
    return atan2f(0.F, __re);
}

}

)ESCAPE";

const std::string complex_half_body = R"ESCAPE(
namespace std {
template <>
struct alignas(2) complex<at::Half> {
  at::Half real_;
  at::Half imag_;

  // Constructors
  complex() = default;

  // implicit casting to and from `complex<float>`.
  // NOTE: computation of `complex<Half>` will occur in `complex<float>`
  __host__ __device__ inline complex(const std::complex<float>& value)
      : real_(value.real()), imag_(value.imag()) {}

  inline __host__ __device__ operator std::complex<float>() const {
    return {real_, imag_};
  }

  at::Half real() const {return real_;}
  at::Half imag() const {return imag_;}

};
}
)ESCAPE";


const std::string &get_complex_body_string() {
  return complex_body;
}

const std::string &get_complex_half_body_string() {
  return complex_half_body;
}

const std::string complex_math = R"ESCAPE(

namespace std {

// norm

template<class _Tp>
inline
_Tp
norm(const complex<_Tp>& __c)
{
    if (isinf(__c.real()))
        return abs(__c.real());
    if (isinf(__c.imag()))
        return abs(__c.imag());
    return __c.real() * __c.real() + __c.imag() * __c.imag();
}

template <class _Tp>
inline
typename __libcpp_complex_overload_traits<_Tp>::_ValueType
norm(_Tp __re)
{
    typedef typename __libcpp_complex_overload_traits<_Tp>::_ValueType _ValueType;
    return static_cast<_ValueType>(__re) * __re;
}

// conj

template<class _Tp>
inline
complex<_Tp>
conj(const complex<_Tp>& __c)
{
    return complex<_Tp>(__c.real(), -__c.imag());
}

template <class _Tp>
inline
typename __libcpp_complex_overload_traits<_Tp>::_ComplexType
conj(_Tp __re)
{
    typedef typename __libcpp_complex_overload_traits<_Tp>::_ComplexType _ComplexType;
    return _ComplexType(__re);
}



// proj

template<class _Tp>
inline
complex<_Tp>
proj(const complex<_Tp>& __c)
{
    complex<_Tp> __r = __c;
    if (isinf(__c.real()) || isinf(__c.imag()))
        __r = complex<_Tp>(INFINITY, copysign(_Tp(0), __c.imag()));
    return __r;
}

template <class _Tp>
inline
typename enable_if
<
    is_floating_point<_Tp>::value,
    typename __libcpp_complex_overload_traits<_Tp>::_ComplexType
>::type
proj(_Tp __re)
{
    if (isinf(__re))
        __re = abs(__re);
    return complex<_Tp>(__re);
}

template <class _Tp>
inline
typename enable_if
<
    is_integral<_Tp>::value,
    typename __libcpp_complex_overload_traits<_Tp>::_ComplexType
>::type
proj(_Tp __re)
{
    typedef typename __libcpp_complex_overload_traits<_Tp>::_ComplexType _ComplexType;
    return _ComplexType(__re);
}

// polar

template<class _Tp>
complex<_Tp>
polar(const _Tp& __rho, const _Tp& __theta = _Tp())
{
    if (isnan(__rho) || signbit(__rho))
        return complex<_Tp>(_Tp(NAN), _Tp(NAN));
    if (isnan(__theta))
    {
        if (isinf(__rho))
            return complex<_Tp>(__rho, __theta);
        return complex<_Tp>(__theta, __theta);
    }
    if (isinf(__theta))
    {
        if (isinf(__rho))
            return complex<_Tp>(__rho, _Tp(NAN));
        return complex<_Tp>(_Tp(NAN), _Tp(NAN));
    }
    _Tp __x = __rho * cos(__theta);
    if (isnan(__x))
        __x = 0;
    _Tp __y = __rho * sin(__theta);
    if (isnan(__y))
        __y = 0;
    return complex<_Tp>(__x, __y);
}

// log

template<class _Tp>
inline
complex<_Tp>
log(const complex<_Tp>& __x)
{
    return complex<_Tp>(log(abs(__x)), arg(__x));
}

// log10

template<class _Tp>
inline
complex<_Tp>
log10(const complex<_Tp>& __x)
{
    return log(__x) / log(_Tp(10));
}

// log2

template<class _Tp>
inline
complex<_Tp>
log2(const complex<_Tp>& __x)
{
    return log(__x) / log(_Tp(2));
}

// sqrt

template<class _Tp>
complex<_Tp>
sqrt(const complex<_Tp>& __x)
{
    if (isinf(__x.imag()))
        return complex<_Tp>(_Tp(INFINITY), __x.imag());
    if (isinf(__x.real()))
    {
        if (__x.real() > _Tp(0))
            return complex<_Tp>(__x.real(), isnan(__x.imag()) ? __x.imag() : copysign(_Tp(0), __x.imag()));
        return complex<_Tp>(isnan(__x.imag()) ? __x.imag() : _Tp(0), copysign(__x.real(), __x.imag()));
    }
    return polar(sqrt(abs(__x)), arg(__x) / _Tp(2));
}

// exp

template<class _Tp>
complex<_Tp>
exp(const complex<_Tp>& __x)
{
    _Tp __i = __x.imag();
    if (__i == 0) {
        return complex<_Tp>(exp(__x.real()), copysign(_Tp(0), __x.imag()));
    }
    if (isinf(__x.real()))
    {
        if (__x.real() < _Tp(0))
        {
            if (!isfinite(__i))
                __i = _Tp(1);
        }
        else if (__i == 0 || !isfinite(__i))
        {
            if (isinf(__i))
                __i = _Tp(NAN);
            return complex<_Tp>(__x.real(), __i);
        }
    }
    _Tp __e = exp(__x.real());
    return complex<_Tp>(__e * cos(__i), __e * sin(__i));
}

// pow

template<class _Tp>
inline
complex<_Tp>
pow(const complex<_Tp>& __x, const complex<_Tp>& __y)
{
    return exp(__y * log(__x));
}

template<class _Tp, class _Up>
inline
complex<typename __promote<_Tp, _Up>::type>
pow(const complex<_Tp>& __x, const complex<_Up>& __y)
{
    typedef complex<typename __promote<_Tp, _Up>::type> result_type;
    return std::pow(result_type(__x), result_type(__y));
}

template<class _Tp, class _Up>
inline
typename enable_if
<
    is_arithmetic<_Up>::value,
    complex<typename __promote<_Tp, _Up>::type>
>::type
pow(const complex<_Tp>& __x, const _Up& __y)
{
    typedef complex<typename __promote<_Tp, _Up>::type> result_type;
    return std::pow(result_type(__x), result_type(__y));
}

template<class _Tp, class _Up>
inline
typename enable_if
<
    is_arithmetic<_Tp>::value,
    complex<typename __promote<_Tp, _Up>::type>
>::type
pow(const _Tp& __x, const complex<_Up>& __y)
{
    typedef complex<typename __promote<_Tp, _Up>::type> result_type;
    return std::pow(result_type(__x), result_type(__y));
}

// __sqr, computes pow(x, 2)

template<class _Tp>
inline
complex<_Tp>
__sqr(const complex<_Tp>& __x)
{
    return complex<_Tp>((__x.real() - __x.imag()) * (__x.real() + __x.imag()),
                        _Tp(2) * __x.real() * __x.imag());
}

// asinh

template<class _Tp>
complex<_Tp>
asinh(const complex<_Tp>& __x)
{
    const _Tp __pi(atan2(+0., -0.));
    if (isinf(__x.real()))
    {
        if (isnan(__x.imag()))
            return __x;
        if (isinf(__x.imag()))
            return complex<_Tp>(__x.real(), copysign(__pi * _Tp(0.25), __x.imag()));
        return complex<_Tp>(__x.real(), copysign(_Tp(0), __x.imag()));
    }
    if (isnan(__x.real()))
    {
        if (isinf(__x.imag()))
            return complex<_Tp>(__x.imag(), __x.real());
        if (__x.imag() == 0)
            return __x;
        return complex<_Tp>(__x.real(), __x.real());
    }
    if (isinf(__x.imag()))
        return complex<_Tp>(copysign(__x.imag(), __x.real()), copysign(__pi/_Tp(2), __x.imag()));
    complex<_Tp> __z = log(__x + sqrt(__sqr(__x) + _Tp(1)));
    return complex<_Tp>(copysign(__z.real(), __x.real()), copysign(__z.imag(), __x.imag()));
}

// acosh

template<class _Tp>
complex<_Tp>
acosh(const complex<_Tp>& __x)
{
    const _Tp __pi(atan2(+0., -0.));
    if (isinf(__x.real()))
    {
        if (isnan(__x.imag()))
            return complex<_Tp>(abs(__x.real()), __x.imag());
        if (isinf(__x.imag()))
        {
            if (__x.real() > 0)
                return complex<_Tp>(__x.real(), copysign(__pi * _Tp(0.25), __x.imag()));
            else
                return complex<_Tp>(-__x.real(), copysign(__pi * _Tp(0.75), __x.imag()));
        }
        if (__x.real() < 0)
            return complex<_Tp>(-__x.real(), copysign(__pi, __x.imag()));
        return complex<_Tp>(__x.real(), copysign(_Tp(0), __x.imag()));
    }
    if (isnan(__x.real()))
    {
        if (isinf(__x.imag()))
            return complex<_Tp>(abs(__x.imag()), __x.real());
        return complex<_Tp>(__x.real(), __x.real());
    }
    if (isinf(__x.imag()))
        return complex<_Tp>(abs(__x.imag()), copysign(__pi/_Tp(2), __x.imag()));
    complex<_Tp> __z = log(__x + sqrt(__sqr(__x) - _Tp(1)));
    return complex<_Tp>(copysign(__z.real(), _Tp(0)), copysign(__z.imag(), __x.imag()));
}

// atanh

template<class _Tp>
complex<_Tp>
atanh(const complex<_Tp>& __x)
{
    const _Tp __pi(atan2(+0., -0.));
    if (isinf(__x.imag()))
    {
        return complex<_Tp>(copysign(_Tp(0), __x.real()), copysign(__pi/_Tp(2), __x.imag()));
    }
    if (isnan(__x.imag()))
    {
        if (isinf(__x.real()) || __x.real() == 0)
            return complex<_Tp>(copysign(_Tp(0), __x.real()), __x.imag());
        return complex<_Tp>(__x.imag(), __x.imag());
    }
    if (isnan(__x.real()))
    {
        return complex<_Tp>(__x.real(), __x.real());
    }
    if (isinf(__x.real()))
    {
        return complex<_Tp>(copysign(_Tp(0), __x.real()), copysign(__pi/_Tp(2), __x.imag()));
    }
    if (abs(__x.real()) == _Tp(1) && __x.imag() == _Tp(0))
    {
        return complex<_Tp>(copysign(_Tp(INFINITY), __x.real()), copysign(_Tp(0), __x.imag()));
    }
    complex<_Tp> __z = log((_Tp(1) + __x) / (_Tp(1) - __x)) / _Tp(2);
    return complex<_Tp>(copysign(__z.real(), __x.real()), copysign(__z.imag(), __x.imag()));
}

// sinh

template<class _Tp>
complex<_Tp>
sinh(const complex<_Tp>& __x)
{
    if (isinf(__x.real()) && !isfinite(__x.imag()))
        return complex<_Tp>(__x.real(), _Tp(NAN));
    if (__x.real() == 0 && !isfinite(__x.imag()))
        return complex<_Tp>(__x.real(), _Tp(NAN));
    if (__x.imag() == 0 && !isfinite(__x.real()))
        return __x;
    return complex<_Tp>(sinh(__x.real()) * cos(__x.imag()), cosh(__x.real()) * sin(__x.imag()));
}

// cosh

template<class _Tp>
complex<_Tp>
cosh(const complex<_Tp>& __x)
{
    if (isinf(__x.real()) && !isfinite(__x.imag()))
        return complex<_Tp>(abs(__x.real()), _Tp(NAN));
    if (__x.real() == 0 && !isfinite(__x.imag()))
        return complex<_Tp>(_Tp(NAN), __x.real());
    if (__x.real() == 0 && __x.imag() == 0)
        return complex<_Tp>(_Tp(1), __x.imag());
    if (__x.imag() == 0 && !isfinite(__x.real()))
        return complex<_Tp>(abs(__x.real()), __x.imag());
    return complex<_Tp>(cosh(__x.real()) * cos(__x.imag()), sinh(__x.real()) * sin(__x.imag()));
}

// tanh

template<class _Tp>
complex<_Tp>
tanh(const complex<_Tp>& __x)
{
    if (isinf(__x.real()))
    {
        if (!isfinite(__x.imag()))
            return complex<_Tp>(copysign(_Tp(1), __x.real()), _Tp(0));
        return complex<_Tp>(copysign(_Tp(1), __x.real()), copysign(_Tp(0), sin(_Tp(2) * __x.imag())));
    }
    if (isnan(__x.real()) && __x.imag() == 0)
        return __x;
    _Tp __2r(_Tp(2) * __x.real());
    _Tp __2i(_Tp(2) * __x.imag());
    _Tp __d(cosh(__2r) + cos(__2i));
    _Tp __2rsh(sinh(__2r));
    if (isinf(__2rsh) && isinf(__d))
        return complex<_Tp>(__2rsh > _Tp(0) ? _Tp(1) : _Tp(-1),
                            __2i > _Tp(0) ? _Tp(0) : _Tp(-0.));
    return  complex<_Tp>(__2rsh/__d, sin(__2i)/__d);
}

// asin

template<class _Tp>
complex<_Tp>
asin(const complex<_Tp>& __x)
{
    complex<_Tp> __z = asinh(complex<_Tp>(-__x.imag(), __x.real()));
    return complex<_Tp>(__z.imag(), -__z.real());
}

// acos

template<class _Tp>
complex<_Tp>
acos(const complex<_Tp>& __x)
{
    const _Tp __pi(atan2(+0., -0.));
    if (isinf(__x.real()))
    {
        if (isnan(__x.imag()))
            return complex<_Tp>(__x.imag(), __x.real());
        if (isinf(__x.imag()))
        {
            if (__x.real() < _Tp(0))
                return complex<_Tp>(_Tp(0.75) * __pi, -__x.imag());
            return complex<_Tp>(_Tp(0.25) * __pi, -__x.imag());
        }
        if (__x.real() < _Tp(0))
            return complex<_Tp>(__pi, signbit(__x.imag()) ? -__x.real() : __x.real());
        return complex<_Tp>(_Tp(0), signbit(__x.imag()) ? __x.real() : -__x.real());
    }
    if (isnan(__x.real()))
    {
        if (isinf(__x.imag()))
            return complex<_Tp>(__x.real(), -__x.imag());
        return complex<_Tp>(__x.real(), __x.real());
    }
    if (isinf(__x.imag()))
        return complex<_Tp>(__pi/_Tp(2), -__x.imag());
    if (__x.real() == 0 && (__x.imag() == 0 || isnan(__x.imag())))
        return complex<_Tp>(__pi/_Tp(2), -__x.imag());
    complex<_Tp> __z = log(__x + sqrt(__sqr(__x) - _Tp(1)));
    if (signbit(__x.imag()))
        return complex<_Tp>(abs(__z.imag()), abs(__z.real()));
    return complex<_Tp>(abs(__z.imag()), -abs(__z.real()));
}

// atan

template<class _Tp>
complex<_Tp>
atan(const complex<_Tp>& __x)
{
    complex<_Tp> __z = atanh(complex<_Tp>(-__x.imag(), __x.real()));
    return complex<_Tp>(__z.imag(), -__z.real());
}

// sin

template<class _Tp>
complex<_Tp>
sin(const complex<_Tp>& __x)
{
    complex<_Tp> __z = sinh(complex<_Tp>(-__x.imag(), __x.real()));
    return complex<_Tp>(__z.imag(), -__z.real());
}

// cos

template<class _Tp>
inline
complex<_Tp>
cos(const complex<_Tp>& __x)
{
    return cosh(complex<_Tp>(-__x.imag(), __x.real()));
}

// tan

template<class _Tp>
complex<_Tp>
tan(const complex<_Tp>& __x)
{
    complex<_Tp> __z = tanh(complex<_Tp>(-__x.imag(), __x.real()));
    return complex<_Tp>(__z.imag(), -__z.real());
}

// Literal suffix for complex number literals [complex.literals]
inline namespace literals
{
  inline namespace complex_literals
  {
    constexpr complex<double> operator""i(long double __im)
    {
        return { 0.0, static_cast<double>(__im) };
    }

    constexpr complex<double> operator""i(unsigned long long __im)
    {
        return { 0.0, static_cast<double>(__im) };
    }


    constexpr complex<float> operator""if(long double __im)
    {
        return { 0.0f, static_cast<float>(__im) };
    }

    constexpr complex<float> operator""if(unsigned long long __im)
    {
        return { 0.0f, static_cast<float>(__im) };
    }
  } // namespace complex_literals
} // namespace literals

} // namespace std

)ESCAPE";

const std::string &get_complex_math_string() {
  return complex_math;
}

} // namespace at::zoom