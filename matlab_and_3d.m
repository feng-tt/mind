clear;clc

% 网格绘图 mesh函数 x:n y:m z:m*n
% fmesh  syms x y
X=[1,2,4]
Y=[3,5]
Z=[4,8,10;5,9,13]
mesh(X,Y,Z)
hidden off % 关闭遮挡 
alpha(0.8) % 设置遮挡透明度
xlabel('X轴');ylabel('Y轴');zlabel('Z轴');

% z=x^2+y^2
n=11;
tem=linspace(0,5,n);
x=repmat(tem,n,1);
y=repmat(tem',1,n);
z=x.^2-y.^2;
mesh(x,y,z)
xlabel('X轴');ylabel('Y轴');zlabel('Z轴');
axis vis3d


% z=sin(sqrt(x^2+y^2))/sqrt(x^2+y^2)
[x,y]=meshgrid(-5:0.5:5);
tem=sqrt(x.^2+y.^2)+eps;
z=sin(tem)./tem;
mesh(x,y,z)
xlabel('X轴');ylabel('Y轴');zlabel('Z轴');
axis vis3d


% meshc函数 绘制等高线
meshc(x,y,z)
xlabel('X轴');ylabel('Y轴');zlabel('Z轴');
axis vis3d
% mechz函数 绘制曲面底座
meshz(x,y,z)
xlabel('X轴');ylabel('Y轴');zlabel('Z轴');
axis vis3d

% surf 绘制曲面图
subplot(1,2,1)
surf(x,y,z)
xlabel('X轴');ylabel('Y轴');zlabel('Z轴');
axis vis3d
title('surf')

subplot(1,2,2)
mesh(x,y,z)
xlabel('X轴');ylabel('Y轴');zlabel('Z轴');
axis vis3d
title('mesh')
% axis([0,5,0,5,-inf,+inf])设置刻度范围
% surfc 绘制等高线
% surfl 添加光线效果 更好看点
% 处理色彩效果：shading
% faceted 默认
% flat 去掉网格线
% interp 色彩平滑


% contour函数 绘制等高线图
[x,y]=meshgrid(-3:0.1:3);
z=3*(1-x).^2.*exp(-(x.^2)-(y+1).^2)...
    -10*(x/5-x.^3-y.^5).*exp(-x.^2-y.^2)...
    -1/3*exp(-(x+1).^2-y.^2);
% contour(x,y,z)
mesh(x,y,z)
% maxz = max(max(z));
% minz = min(min(z));
% levels = linspace(minz,maxz,10);
% contour(x,y,z,n,'LineWidth',2，'--','ShowText','on',levels)
% 设置等高线层数为n,线宽2，线条虚线，显示高度,设置固定n个等高线
xlabel('X轴');ylabel('Y轴');zlabel('Z轴');
% contourf带有颜色填充的等高线
% contour3三维等高线

interval=[-5 5 -5 5 0 5];
f=@(x,y,z)x.^2+y.^2-z.^2;
fimplicit3(f,interval)
clear;clc