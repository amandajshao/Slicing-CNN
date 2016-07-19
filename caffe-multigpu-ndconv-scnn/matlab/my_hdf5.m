%% generate hdf5 of ucf101
clc;clear;

path_root = '/home/jshao/jshao/project_action/THUMOS_UCF101/';
path_source = [path_root,'feature_maps/vgg_16_rgb_action_feature_map/conv4_3_feature_map_all/'];
% path_source_dir = dir([path_source,'*.mat']);
% [list_total_samples,list_lens,list_labs] = textread([path_root,'ucf-101_file_list_all_test.txt'],'%s %d %d');
[list_total_samples,list_lens,list_labs] = textread([path_root,'temp.txt'],'%s %d %d');

num_total_samples = length(list_total_samples);
chunksz = 2;
created_flag = false;
totalct = 0;
sample_stride = 2;
compress_fil_sz = 4;

file_hdf = [path_root,'trial.h5'];
% CREATE list.txt containing filename, to be used as source for HDF5_DATA_LAYER
FILE = fopen([path_root,'list.txt'], 'w');
fprintf(FILE, '%s', file_hdf);
fclose(FILE);

for batchno = 1 : num_total_samples/chunksz
    fprintf('batch no. %d\n', batchno);
    last_read = (batchno-1)*chunksz;
    
    tic;
    
    batchdata = zeros(40,30,100,512,chunksz); % [width,height,time,channels,num_sample]
    batchlabs = zeros(1,chunksz);
    for i = 1 : chunksz
        file_name = list_total_samples{last_read+i};
        file_name_ori = file_name(find(file_name=='/')+1:end);
        disp([num2str(i),':',file_name_ori]);
        batchdata_cur = importdata([path_source,file_name_ori,'.mat']);
        batchlabs_cur = list_labs(last_read+i);
        % transpose to [width,height,time,channels]
        batchdata_cur = permute(batchdata_cur,[4 3 2 1]);
        % sample time by stride(=2)
        batchdata_cur = batchdata_cur(:,:,1:sample_stride:end,:);
        % pad zeros to fixed length of 100
        sz = size(batchdata_cur);
        if sz(3) ~= 100
            pad_zero = zeros(sz(1),sz(2),100-sz(3),sz(4));
            batchdata_cur = cat(3, batchdata_cur, pad_zero);
        end
        batchdata(:,:,:,:,i) = batchdata_cur;
        batchlabs(:,i) = batchlabs_cur;
    end
    
    startloc = struct('dat',[1,1,1,1,totalct+1],'lab',[1,totalct+1]);
    curr_dat_sz = store2hdf5(file_hdf, batchdata, batchlabs, ~created_flag, ...
        startloc, chunksz, compress_fil_sz);
    created_flag = true;
    totalct = curr_dat_sz(end);
    
    t2=toc;
end

% display structure of the stored HDF5 file
h5disp(file_hdf);

% read
data_rd = h5read(file_hdf, '/data');
label_rd = h5read(file_hdf, '/label');













