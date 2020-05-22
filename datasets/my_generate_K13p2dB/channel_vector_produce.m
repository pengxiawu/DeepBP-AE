% produce channel matrix, in which the rows indicate the sample index,
% columns are antenna index
close all
clc
clear

%% mmWave spatial domain channel vectors.
num_antennas = 256; % antenna number
num_paths = 3;   % path number
Wavelenth = 1; % Wavelength
d = Wavelenth/2 ; % antenna spacing
num_sam = 20000;
sparsity=3;
% dense spatial channel vectors
H_org = spatial_channel(num_antennas,num_sam,num_paths) + 0.* randn(num_antennas, num_sam); 

randseed(43)

%% sparse beamspace channel vectors
% deta = 1/num_antennas;
% for i = -(num_antennas-1)/2:1:(num_antennas-1)/2
%     U(:,i+(num_antennas+1)/2) = (1/num_antennas)*exp(1i*[0:num_antennas-1]*2*pi*deta*i).';
% end
%H_beam = U.'* H_org; % beamspace channel
H_beam =  (1/num_antennas).* fft(H_org,num_antennas);

H_beam_sparsity = zeros(size(H_beam));
H_beam_sparsity_syn = zeros(num_antennas, num_sam*2);
H_beam_module = abs(H_beam);
for sample = 1:1:num_sam
[val,index] = sort(H_beam_module(:, sample),'descend');
H_beam_sparsity(index(1:sparsity), sample) = H_beam(index(1:sparsity), sample);
H_beam_sparsity_syn(:,sample*2-1) = real(H_beam_sparsity(:, sample));
H_beam_sparsity_syn(:,sample*2) = imag(H_beam_sparsity(:, sample));
end


% save data
save('H_beam_sparsity_syn3.mat','H_beam_sparsity_syn')






