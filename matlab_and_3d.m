clear;clc

% �����ͼ mesh���� x:n y:m z:m*n
% fmesh  syms x y
X=[1,2,4]
Y=[3,5]
Z=[4,8,10;5,9,13]
mesh(X,Y,Z)
hidden off % �ر��ڵ� 
alpha(0.8) % �����ڵ�͸����
xlabel('X��');ylabel('Y��');zlabel('Z��');

% z=x^2+y^2
n=11;
tem=linspace(0,5,n);
x=repmat(tem,n,1);
y=repmat(tem',1,n);
z=x.^2-y.^2;
mesh(x,y,z)
xlabel('X��');ylabel('Y��');zlabel('Z��');
axis vis3d


% z=sin(sqrt(x^2+y^2))/sqrt(x^2+y^2)
[x,y]=meshgrid(-5:0.5:5);
tem=sqrt(x.^2+y.^2)+eps;
z=sin(tem)./tem;
mesh(x,y,z)
xlabel('X��');ylabel('Y��');zlabel('Z��');
axis vis3d


% meshc���� ���Ƶȸ���
meshc(x,y,z)
xlabel('X��');ylabel('Y��');zlabel('Z��');
axis vis3d
% mechz���� �����������
meshz(x,y,z)
xlabel('X��');ylabel('Y��');zlabel('Z��');
axis vis3d

% surf ��������ͼ
subplot(1,2,1)
surf(x,y,z)
xlabel('X��');ylabel('Y��');zlabel('Z��');
axis vis3d
title('surf')

subplot(1,2,2)
mesh(x,y,z)
xlabel('X��');ylabel('Y��');zlabel('Z��');
axis vis3d
title('mesh')
% axis([0,5,0,5,-inf,+inf])���ÿ̶ȷ�Χ
% surfc ���Ƶȸ���
% surfl ��ӹ���Ч�� ���ÿ���
% ����ɫ��Ч����shading
% faceted Ĭ��
% flat ȥ��������
% interp ɫ��ƽ��


% contour���� ���Ƶȸ���ͼ
[x,y]=meshgrid(-3:0.1:3);
z=3*(1-x).^2.*exp(-(x.^2)-(y+1).^2)...
    -10*(x/5-x.^3-y.^5).*exp(-x.^2-y.^2)...
    -1/3*exp(-(x+1).^2-y.^2);
% contour(x,y,z)
mesh(x,y,z)
% maxz = max(max(z));
% minz = min(min(z));
% levels = linspace(minz,maxz,10);
% contour(x,y,z,n,'LineWidth',2��'--','ShowText','on',levels)
% ���õȸ��߲���Ϊn,�߿�2���������ߣ���ʾ�߶�,���ù̶�n���ȸ���
xlabel('X��');ylabel('Y��');zlabel('Z��');
% contourf������ɫ���ĵȸ���
% contour3��ά�ȸ���

interval=[-5 5 -5 5 0 5];
f=@(x,y,z)x.^2+y.^2-z.^2;
fimplicit3(f,interval)
clear;clc