filename='batchSize';

path=strcat(filename,'.dat');
opath=strcat('fig/', filename, '.jpg');

X = load(path);
x=X(:,1); y= X(:,2);

fig = plot(x,y, 'ro-');
xlabel(filename);
ylabel('validation score');
ylim([0 1]);
saveas(fig, opath)
    
