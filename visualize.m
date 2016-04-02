directory = dir('imagecsv');
names = {directory.name};
mymap = jet;
mymap(1, : ) = 1;
for i = 4: size(names, 2)
    title = strcat('imagecsv/', names(i));
    image = csvread(title{1});
    figure;
    imagesc(image);
    colormap(mymap);
    set(gca, 'xtick', [])
    set(gca, 'ytick', [])
    colorbar
    event = names(i);
    title = strsplit(event{1}, '.');
    location = strcat('images/', title{1});
    savefig(location)
end