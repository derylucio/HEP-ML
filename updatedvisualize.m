none_dir = dir('nonecsv');
none = {none_dir.name};
bjets_dir = dir('bjetcsv');
bjets= {bjets_dir.name};
bbellipse_dir = dir('bbellipsecsv');
bbellipse = {bbellipse_dir.name};
vbf_dir = dir('vbfcsv');
vbf = {vbf_dir.name};
mymap = jet;
mymap(1, : ) = 1;
for i = 4: size(vbf, 2)
    clear title
    figure(1)
    ptitle = strcat('nonecsv/', none(i - 1));
    image = csvread(ptitle{1});
    subplot(2, 2, 1)
    imagesc(image')
    colorbar
    set(gca, 'xtick' , 0:10:50)
    set(gca, 'xticklabel', {'-2.5', '-1.5', '-0.5', '0.5', '1.5', '2.5'})
    set(gca, 'ytick' , 0:12.6:63)
    set(gca, 'yticklabel', {'3.15', '1.89', '0.63', '-0.63', '-1.89', '-3.15'})
    xlabel('eta')
    ylabel('phi')
    title('Background')
    
    ptitle = strcat('bjetcsv/', bjets(i));
    image = csvread(ptitle{1});
    subplot(2, 2, 2)
    imagesc(image') 
    colorbar
    set(gca, 'xtick' , 0:10:50)
    set(gca, 'xticklabel', {'-2.5', '-1.5', '-0.5', '0.5', '1.5', '2.5'})
    set(gca, 'ytick' , 0:12.6:63)
    set(gca, 'yticklabel', {'3.15', '1.89', '0.63', '-0.63', '-1.89', '-3.15'})
    xlabel('eta')
    ylabel('phi')
    title('BJet')

    
    ptitle = strcat('bbellipsecsv/', bbellipse(i));
    image = csvread(ptitle{1});
    subplot(2, 2, 3)
    imagesc(image')
    colorbar
    set(gca, 'xtick' , 0:10:50)
    set(gca, 'xticklabel', {'-2.5', '-1.5', '-0.5', '0.5', '1.5', '2.5'})
    set(gca, 'ytick' , 0:12.6:63)
    set(gca, 'yticklabel', {'3.15', '1.89', '0.63', '-0.63', '-1.89', '-3.15'})
    xlabel('eta')
    ylabel('phi')
    title('BBellipse')

    
    ptitle = strcat('vbfcsv/', vbf(i));
    image = csvread(ptitle{1});
    subplot(2, 2, 4)
    imagesc(image')
    colormap(mymap)
    colorbar
    set(gca, 'xtick' , 0:10:50)
    set(gca, 'xticklabel', {'-2.5', '-1.5', '-0.5', '0.5', '1.5', '2.5'})
    set(gca, 'ytick' , 0:12.6:63)
    set(gca, 'yticklabel', {'3.15', '1.89', '0.63', '-0.63', '-1.89', '-3.15'})
    xlabel('eta')
    ylabel('phi')
    title('VBFJet')
    
    event = none(i - 1);
    title = strsplit(event{1}, '.');
    location = strcat('QuadPlots/', title{1});
    savefig(location)
end