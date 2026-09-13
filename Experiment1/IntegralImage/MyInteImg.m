function MyInteImg_result=MyInteImg(img)

%���ƻ���ͼ��ļ������
% ����ͼ��ʽ��SA(i,j) = sum_{x<=i, y<=j} img(x,y)
% �ɵݹ�ʵ�֣�SA(i,j) = img(i,j) + SA(i-1,j) + SA(i,j-1) - SA(i-1,j-1)

[height, width] = size(img);
MyInteImg_result = double(zeros(height, width));

% ��һ�е�һ�е��ۼ�
MyInteImg_result(1, 1) = double(img(1, 1));

% ��һ�е��ۼӣ�ֻ�ܴ��������
for j = 2:width
    MyInteImg_result(1, j) = MyInteImg_result(1, j-1) + double(img(1, j));
end

% ��һ�е��ۼӣ�ֻ�ܴ��ϱ�����
for i = 2:height
    MyInteImg_result(i, 1) = MyInteImg_result(i-1, 1) + double(img(i, 1));
end

% һ��λ�ã��ݹ鹫ʽ
for i = 2:height
    for j = 2:width
        MyInteImg_result(i, j) = double(img(i, j)) ...
            + MyInteImg_result(i-1, j) ...
            + MyInteImg_result(i, j-1) ...
            - MyInteImg_result(i-1, j-1);
    end
end

%����ԭͼ�����õ��Ļ���ͼ��MyInteImg_result
end
