function DisTrans_result=MyDisTrans(bw)

%���ƾ���任�ļ������
% ʹ������ɨ��� chamfer �����㷨�������پ��룩
% ��һ�飺�����ϵ�����ɨ��
% �ڶ��飺�����µ�����ɨ��

[height, width] = size(bw);
DisTrans_result = double(bw);

% ��һ��ɨ�裺���ϵ�����
for i = 1:height
    for j = 1:width
        if bw(i, j) == 1
            DisTrans_result(i, j) = 0;
        else
            neighbors = [];
            if i > 1,       neighbors = [neighbors, DisTrans_result(i-1, j)]; end
            if j > 1,       neighbors = [neighbors, DisTrans_result(i, j-1)]; end
            if i > 1 && j > 1,   neighbors = [neighbors, DisTrans_result(i-1, j-1)]; end
            if i > 1 && j < width, neighbors = [neighbors, DisTrans_result(i-1, j+1)]; end
            if isempty(neighbors)
                DisTrans_result(i, j) = inf;
            else
                DisTrans_result(i, j) = 1 + min(neighbors);
            end
        end
    end
end

% �ڶ���ɨ�裺���µ�����
for i = height:-1:1
    for j = width:-1:1
        if bw(i, j) ~= 1
            neighbors = [];
            if i < height,       neighbors = [neighbors, DisTrans_result(i+1, j)]; end
            if j < width,        neighbors = [neighbors, DisTrans_result(i, j+1)]; end
            if i < height && j < width,  neighbors = [neighbors, DisTrans_result(i+1, j+1)]; end
            if i < height && j > 1,      neighbors = [neighbors, DisTrans_result(i+1, j-1)]; end
            if ~isempty(neighbors)
                DisTrans_result(i, j) = min(DisTrans_result(i, j), 1 + min(neighbors));
            end
        end
    end
end

maxVal = max(DisTrans_result(:));
if maxVal > 0
    DisTrans_result = DisTrans_result / maxVal;
end

%����ԭ��ֵͼ�񾭹�����任��Ľ��DisTrans_result
end
