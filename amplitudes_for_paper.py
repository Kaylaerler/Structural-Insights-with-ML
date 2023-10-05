import numpy as np
import preprocess_data
import pickle
import preprocess_data as ppd
import post_process as post
signals_dictionary = ppd.load_data_set('preprocessed_data', save = False, load = True)
labels = signals_dictionary[0]['labels']

disp_max = []
disp_min = []
vel_max = []
vel_min = []
acc_max = []
acc_min = []
v_force_max = []
v_force_min = []

print('-------------------------')
disp_idx = 1
print('displacement index label: ' + labels[disp_idx]['name'] + ', Units:'+ labels[disp_idx]['units'])
vel_idx = 2
print('velocity index label: ' + labels[vel_idx]['name'] + ', Units:'+ labels[vel_idx]['units'])
acc_idx = 3
print('accleration index label: ' + labels[acc_idx]['name'] + ', Units:'+ labels[acc_idx]['units'])
out_force_idx = 4
print('Outrigger index label: ' + labels[out_force_idx]['name'] + ', Units:'+ labels[out_force_idx]['units'])
act_force_idx = 5
print('Actuator index label: ' + labels[act_force_idx]['name'] + ', Units:'+ labels[act_force_idx]['units'])
print('-------------------------')

for keys in signals_dictionary:

    displacement = signals_dictionary[keys]['data'][:, disp_idx]
    disp_max.append(np.max(np.abs(displacement)))

    velocity = signals_dictionary[keys]['data'][:, vel_idx]
    vel_max.append(np.max(np.abs(velocity)))

    acceleration = signals_dictionary[keys]['data'][:, acc_idx]
    acc_max.append(np.max(np.abs(acceleration)))    

    out_force = signals_dictionary[keys]['data'][:, out_force_idx]
    act_force = signals_dictionary[keys]['data'][:, act_force_idx]
    v_force   = out_force + act_force
    v_force_max.append(np.max(v_force))

min_index = np.where(disp_max == np.max(disp_max))[0][0]
print(min_index)
post.plot_signals(min_index, preprocessed_data_directory = 'preprocessed_data')

# convert to metric units
in_to_mm = 25.4
g = 386.1 #in/s^2
ton_to_kN = 8.89644
max_amp_disp = np.max(disp_max)*in_to_mm
min_amp_disp = np.min(disp_max)*in_to_mm
max_amp_vel = np.max(vel_max)*in_to_mm
min_amp_vel = np.min(vel_max)*in_to_mm
max_amp_acc = np.max(acc_max)
min_amp_acc = np.min(acc_max)
max_amp_v_force = np.max(v_force_max)*ton_to_kN
min_amp_v_force = np.min(v_force_max)*ton_to_kN

print(f'Disp range between {min_amp_disp:.2f} and {max_amp_disp:.2f} mm')
print(f'Vel range between {min_amp_vel:.2f} and {max_amp_vel:.2f} mm/s')
print(f'Acc range between {min_amp_acc:.10f} and {max_amp_acc:.4f} g')
print(f'V force range between {min_amp_v_force:.2f} and {max_amp_v_force:.2f} kN')
