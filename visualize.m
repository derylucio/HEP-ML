image = csvread('imagecsv/ggf6383.csv');
figure;
imagesc(image);
mymap = jet;
for i = 1:3
    mymap(1,i) = 1;
end
colormap(mymap);
set(gca, 'xtick', [])
set(gca, 'ytick', [])
colorbar
savefig('images/ggf6383')