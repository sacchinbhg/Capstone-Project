import matplotlib.pyplot as plt
import numpy as np
import csv
from params import par

pose_GT_dir = '/home/sacchin/Desktop/dnt/capstone/DeepVO-pytorch/KITTI/pose_GT/'
predicted_result_dir = '/home/sacchin/Desktop/dnt/capstone/result/'
gradient_color = True

def plot_route(gt, out):
    x_idx = 3
    y_idx = 5

    # Ground Truth plot with green color
    x = [v for v in gt[:, x_idx]]
    y = [v for v in gt[:, y_idx]]
    plt.plot(x, y, color='g', label='Ground Truth')

    # LSTM_VO plot with red color
    x = [v for v in out[:, x_idx]]
    y = [v for v in out[:, y_idx]]
    plt.plot(x, y, color='r', label='LSTM_VO')

    # Set equal aspect ratio
    plt.gca().set_aspect('equal', adjustable='datalim')

# video_list = ['00', '02', '08', '06', '01']
video_list = ['01', '03', '04', '05', '07', '09', '10']

for video in video_list:
    print('='*50)
    print('Video {}'.format(video))

    GT_pose_path = '{}{}.npy'.format(pose_GT_dir, video)
    gt = np.load(GT_pose_path)
    pose_result_path = '{}out_{}.txt'.format(predicted_result_dir, video)
    with open(pose_result_path) as f_out:
        out = [l.split('\n')[0] for l in f_out.readlines()]
        for i, line in enumerate(out):
            out[i] = [float(v) for v in line.split(',')]
        out = np.array(out)
        mse_rotate = 100 * np.mean((out[:, :3] - gt[:, :3])**2)
        mse_translate = np.mean((out[:, 3:] - gt[:, 3:6])**2)
        print('mse_rotate: ', mse_rotate)
        print('mse_translate: ', mse_translate)

    # Write the CSV file
    csv_filename = '{}trajectory_{}.csv'.format(predicted_result_dir, video)
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Ground Truth X', 'Ground Truth Y', 'Predicted X', 'Predicted Y'])
        for gt_pose, pred_pose in zip(gt, out):
            writer.writerow([gt_pose[3], gt_pose[5], pred_pose[3], pred_pose[5]])
    print(f"CSV file for video {video} saved as {csv_filename}")

    # Plotting section (optional based on your requirements)
    if gradient_color:
        step = 200
        plt.clf()
        plt.scatter([gt[0][3]], [gt[0][5]], label='sequence start', marker='s', color='k')
        for st in range(0, len(out), step):
            end = st + step
            g = max(0.2, st/len(out))
            c_gt = (0, g, 0)
            c_out = (1, g, 0)
            plot_route(gt[st:end], out[st:end])
            if st == 0:
                plt.legend()
            plt.title('Video {}'.format(video))
            save_name = '{}route_{}_gradient.png'.format(predicted_result_dir, video)
        plt.savefig(save_name)
    else:
        plt.clf()
        plt.scatter([gt[0][3]], [gt[0][5]], label='sequence start', marker='s', color='k')
        plot_route(gt, out)
        plt.legend()
        plt.title('Video {}'.format(video))
        save_name = '{}route_{}.png'.format(predicted_result_dir, video)
        plt.savefig(save_name)