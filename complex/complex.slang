import sgl.device.print;

struct Complex
{
    float re;
    float im;

    __init(float real, float imaginary ) { re = real; im = imaginary;}
}

Complex operator+(Complex a, Complex b)
{
    return Complex(a.re + b.re, a.im + b.im);
}

Complex operator-(Complex a, Complex b)
{
    return Complex(a.re - b.re, a.im - b.im);
}

Complex operator*(Complex a, Complex b)
{
    return Complex(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}

Complex operator/(Complex a, Complex b)
{
    return Complex( (a.re*b.re + a.im*b.im) / (b.re*b.re + b.im*b.im), (-a.re*b.im + a.im*b.re) / (b.re*b.re + b.im*b.im));
}

[shader("compute")]
[numthreads(1, 1, 1)]
void test(uint2 tid: SV_DispatchThreadID)
{
    Complex c1 = Complex(1.0, 2.0);
    Complex c2 = Complex(3.0, 4.0);
    
    Complex c3 = c1 + c2;
    Complex c4 = c1 - c2;
    Complex c5 = c1 * c2;
    Complex c6 = c1 / c2;

    print("c1 + c2 = {} + {}j", c3.re, c3.im);
    print("c1 - c2 = {} + {}j", c4.re, c4.im);
    print("c1 * c2 = {} + {}j", c5.re, c5.im);
    print("c1 / c2 = {} + {}j", c6.re, c6.im);
}

// Add two pixels together
Complex add(Complex a, Complex b)
{
    return a + b;
}
