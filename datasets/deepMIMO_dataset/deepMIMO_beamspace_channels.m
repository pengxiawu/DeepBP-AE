
close all
clc
clear

%% access deepMIMO spatial-domain channels
% 'deepMIMO_dataset_mat' is produced by the deepMIMO, online available: https://www.deepmimo.net/
% Ref: <DeepMIMO: A Generic Deep Learning Dataset for Millimeter Wave and
% Massive MIMO Applications> 
load('./DeepMIMO Dataset/DeepMIMO_dataset.mat'); 
num_antennas = 256; % antenna number
num_users = 54481;
deepMIMO_channels = zeros(num_antennas, num_users);
for i = 1:num_users
    deepMIMO_channels(:, i) = DeepMIMO_dataset{1}.user{i}.channel;
end
% save('deepMIMO_channels', 'deepMIMO_channels');

%% transform to sparse beamspace channels
H_org = deepMIMO_channels;
sparsity = 3; 
randseed(43)

deta = 1/num_antennas;
% for i = -(num_antennas-1)/2:1:(num_antennas-1)/2
%     U(:,i+(num_antennas+1)/2) = (1/num_antennas)*exp(1i*[0:num_antennas-1]*2*pi*deta*i).';
% end
% H_beam = U.'* H_org; % beamspace channel
H_beam =  (1/num_antennas).* fft(H_org,num_antennas);

H_beam_sparsity = zeros(size(H_beam));
H_beam_sparsity_syn = zeros(num_antennas, num_users*2);
H_beam_module = abs(H_beam);
for sample = 1:1:num_users
[val,index] = sort(H_beam_module(:, sample),'descend');
H_beam_sparsity(index(1:sparsity), sample) = H_beam(index(1:sparsity), sample);
H_beam_sparsity_syn(:,sample*2-1) = real(H_beam_sparsity(:, sample));
H_beam_sparsity_syn(:,sample*2) = imag(H_beam_sparsity(:, sample));
end

% % save data
save('H_beam_sparsity_syn3.mat','H_beam_sparsity_syn')





