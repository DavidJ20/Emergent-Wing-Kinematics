N = 300;
n = 1:N;
%Matlab callocation points
theta = (pi/(2*N))*(2*n-1);

%Moore callocation points
%theta = (pi*(2*n+1)./(2*(N+1)));
x = (cos(theta));
%f = (2+x).*sqrt(1-x.^2)-(1+2*x).*acos(x);
f = 3.*x.^2; 
%f = cos(x);

% FFT based Spectral Integration
% allocate memory
N=300;
b=zeros(1,N+1);
% DCT
a = dct(f);
a(2:end) = (sqrt(2/N)).*a(2:end);
a(1) = (1/(sqrt(N))).*a(1);

a(1)=a(1)/2;
%a(N+1) = a(N+1)/2;
% Antidifferentiate
b(1)= a(2)/4;
b(2)=a(1)-a(3)/2;
b(3:N-1) = (a(2:N-2)-a(4:N))./[4:2:2*N-4];
%b(N) = a(N-1)/(2*N-2)-a(N+1)/(N*N-1);
%b(N+1) = a(N)/(2*N);


B = b;
pn = zeros(1,N+1);
for j = 2:10
    pn = pn + B(j-1).*chebyshevT(j, x)';
end

% Inverse Transform
% c=[b;b(N:-1:2)];
% c(1)c(1)*2;c(N+1)-c(N+1)*2;
% %remove imaginary part introduced by rounding errors
% F=real(ffl(c))/2;
% % Projection
% FL=F(1:N+1)-ones(N+1,1)*F(N+1); 