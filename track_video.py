from moviepy.editor import VideoFileClip
import mark_lane_lines as mark

def track(file,outfile):
    clip = VideoFileClip(file)\
        # .subclip(20,45)

    marked = clip.fl_image(mark.mark_lane_zone)
    marked.write_videofile(outfile,audio=False)


import time
start = time.time()
src_file = 'project_video.mp4'
out_file='{}_marked_{}.mp4'.format(src_file.split('.')[0],int(time.time()))
track(src_file,out_file)
end = time.time()
print('Cost time: {}'.format(end-start))
print('Complete scan window: ',mark.finder.complete_scan_window)
print('Frame count: ',mark.finder.frame_count)
print('Correct success count:',mark.finder.correct_success)
print('Failed correct count:',mark.finder.fail_correct)