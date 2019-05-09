user = ('dataset5/data_E/');
a =dir(user);


max_size =[272,249,3];
M=279;
N = 249;

for i=3:1:length(a)
    i
    template = zeros(M,N,3);
    filename =a(i).name;
    f = im2double(imread(strcat(user,filename)));
    [m,n,~] = size(f);
    template(floor(1+(M-m)/2):(m+floor((M-m)/2)),floor(1+(N-n)/2):(n+floor((N-n)/2)),:) = f;
    %imshow(im2uint8(template));
    imwrite(im2uint8(template),strcat(user,filename));
end

max_size
    