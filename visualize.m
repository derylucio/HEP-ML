image = csvread('img_matrices/vbf65300.csv');
figure;
imagesc(image);
mymap = jet;
for i = 1:3
    mymap(1,i) = 1;
end
colormap(mymap);
colorbar
