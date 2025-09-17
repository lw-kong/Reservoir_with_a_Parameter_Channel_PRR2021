function zc = func_IHJM(z,a)

zc = a + 0.9 * z * exp( 1i*0.4 - 1i*6/(1+abs(z)^2) );

end

