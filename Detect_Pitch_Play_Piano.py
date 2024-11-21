#************************************************ Import Libraries ************************************************

import numpy as np
from scipy.signal import fftconvolve
import wave
import math
import warnings

import pyaudio

import pygame
import piano_lists as pl

#************************************************** All Functions ************************************************

#------------------------------- YIN Algorithm Functions ------------------------------

def ACF(block, window_size, max_tau):

  # W elements, window size elements are correlated, not the whole signal
  x_wind = block[:window_size]


  y_wind = block[:(window_size+max_tau)]

  y_wind_flipped = y_wind[::-1]


  corr  = fftconvolve(x_wind, y_wind_flipped, mode="full")

  corr_valid = corr[:(window_size+max_tau)]

  corr_all_tau = corr_valid[::-1]

  # all we need is max_tau +1 elements, lags from zero until max_tau(inclusive)

  corr_until_max_tau = corr_all_tau[:max_tau+1]

  return (corr_until_max_tau)


def DF(block, window_size, tau, corr_all_tau):

  corr_0 =  corr_all_tau[0]

  x_wind_tau = block[tau : tau + window_size]

  corr_tau_0 = np.sum(np.power(x_wind_tau,2))


  return ( corr_0 + corr_tau_0 - (2 *corr_all_tau[tau]) )


def CMNDF(block, window_size, tau, corr_all_tau, DF_run_sum):
    if tau == 0:
        return (1, 0)

    DF_tau = DF(block, window_size, tau, corr_all_tau)

    # DF_run_sum is sum until tau-1
    DF_run_sum_new = DF_run_sum + DF_tau

    if DF_run_sum_new !=0:
      CMNDF_result = (tau * DF_tau) / DF_run_sum_new
    else:
      CMNDF_result = 1
       

    return ( CMNDF_result, DF_run_sum_new )


def min_tau_CMNDF(block, window_size,max_tau,threshold):

  # units in tau (discrete)
  corr_all_tau = ACF(block, window_size, max_tau)

  threshold_tau = -1
  minimum_tau = -1
  min_value = -1

  threshold_passed = False


  CMNDF_running_sum = 0

  #max_tau inclusive
  results = np.ones((max_tau+1, 1))

  #max_tau inclusive
  for tau in range (max_tau+1):

    CMNDF_result, CMNDF_running_sum_new = CMNDF(block, window_size, tau, corr_all_tau, CMNDF_running_sum)

      # threshold passed but still on a downward trajectory
      # update until we get the minimum

    if threshold_passed:

      if CMNDF_result <= results[tau-1]:
        # in a downward trajectory
        results[tau] = CMNDF_result
        CMNDF_running_sum = CMNDF_running_sum_new

        threshold_tau = tau
        #print("downward traj: continue even after threshold")

      else:
        break


    else:

      results[tau] = CMNDF_result
      CMNDF_running_sum = CMNDF_running_sum_new

    if CMNDF_result < threshold and (not threshold_passed):

      threshold_tau = tau
      threshold_passed = True


  if threshold_tau ==-1:
    minimum_tau = np.argmin(results)
    min_value = np.min(results)

  return (threshold_tau, minimum_tau,min_value)


def detect_pitch(block, window_size, max_tau,threshold):

  threshold_tau, minimum_tau,min_value = \
  min_tau_CMNDF(block, window_size,max_tau,threshold)

  fundamental_period_samples = None
  confidence = None


  if threshold_tau != -1:
    fundamental_period_samples = threshold_tau
    confidence = threshold
  else:
    if minimum_tau !=0:
      fundamental_period_samples = minimum_tau
      confidence = min_value

    else:
      #print("Error: A false frequency of infinty (minimum tau zero) was reported.)

      fundamental_period_samples = -1
      confidence = 1

  return fundamental_period_samples, confidence


#--------------------------------------------------------------------------------------


#------------------------------ Helper Functions --------------------------------------

def get_FFT_len(Block_len):

  # power of two greater than the minimum
  a = int(math.log2(Block_len))

  if 2**a == Block_len:
    return Block_len

  return 2**(a + 1)


def find_duplicates(arr):
  # Get unique values and their counts
  unique_values, counts = np.unique(arr, return_counts=True)
  
  # Find indices of duplicates
  duplicate_indices = np.where(counts > 1)[0]
  
  # Extract duplicates and their indices
  duplicates = unique_values[duplicate_indices]

  return duplicates, 

#---------------------------------------------------------------------------------------


#------------------------------- Piano GUI Setup Functions -----------------------------

# -----Title Bar Setup Function ------

def draw_title_bar():
    instruction_text = small_font.render(' 1, Press "M" Mic, "R" for Recording', True, 'black')
    screen.blit(instruction_text, (WIDTH - 750, 10))

    instruction_text2 = small_font.render(' 2, Press Space Bar to Start !', True, 'black')
    screen.blit(instruction_text2, (WIDTH - 750, 40))

    Param_text = f'Sampling Rate = {RATE} Hz, Min Frequency = {min_F0} Hz, Max Frequency = {max_F0} Hz'

    instruction_text2 = small_font.render(Param_text, True, 'black')
    screen.blit(instruction_text2, (WIDTH - 600, 70))

    title_text = font.render('Play Any Instrument, Including Singing!', True, 'white')
    screen.blit(title_text, (98, 18))
    title_text = font.render('Play Any Instrument, Including Singing!', True, 'black')
    screen.blit(title_text, (100, 20))


# ----- Draw Piano Keys Function -----

def draw_piano(whites, blacks):
    white_rects = []
    for i in range(52):
        rect = pygame.draw.rect(screen, 'white', [i * 35, HEIGHT - 300, 35, 300], 0, 2)
        white_rects.append(rect)
        pygame.draw.rect(screen, 'black', [i * 35, HEIGHT - 300, 35, 300], 2, 2)
        key_label = small_font.render(white_notes[i], True, 'black')
        screen.blit(key_label, (i * 35 + 3, HEIGHT - 20))
    skip_count = 0
    last_skip = 2
    skip_track = 2
    black_rects = []
    for i in range(36):
        rect = pygame.draw.rect(screen, 'black', [23 + (i * 35) + (skip_count * 35), HEIGHT - 300, 24, 200], 0, 2)
        for q in range(len(blacks)):
            if blacks[q][0] == i:
                if blacks[q][1] > 0:
                    pygame.draw.rect(screen, 'deeppink3', [23 + (i * 35) + (skip_count * 35), HEIGHT - 300, 24, 200], 10, 2)
                    blacks[q][1] -= 1

        key_label = real_small_font.render(black_notes[i], True, 'white')
        screen.blit(key_label, (25 + (i * 35) + (skip_count * 35), HEIGHT - 120))
        black_rects.append(rect)
        skip_track += 1
        if last_skip == 2 and skip_track == 3:
            last_skip = 3
            skip_track = 0
            skip_count += 1
        elif last_skip == 3 and skip_track == 2:
            last_skip = 2
            skip_track = 0
            skip_count += 1

    for i in range(len(whites)):
        if whites[i][1] > 0:
            j = whites[i][0]
            pygame.draw.rect(screen, 'darkmagenta', [j * 35, HEIGHT - 100, 35, 100], 16, 2)
            whites[i][1] -= 1

    return white_rects, black_rects, whites, blacks

#-------------------------------------------------------------------------------------------

#*****************************************************************************************************************************


#************************************************** Tunable Hyper Parameters **************************************************


# These hyper-parameters can be tuned 

# Sampling Frequency
#RATE = 11025         # Hz frames per second
RATE = 44100         # Hz frames per second

# The maximum rate we can have while keeping an FFT length of 2048
#RATE = 18000


min_F0 = 27         # Minimum detectable frequency in Hz

threshold = 0.1     # Confidence threshold to report the fundamental period
                    # Values close to zero mean stricter requirement 


#************************************************** Fixed Hyper Parameters **************************************************
tau_min = 4
max_F0 = (1/tau_min)*RATE  # Maximum detectable frequency in Hz

# By nyquist The max_F0 detectable is 0.5*RATE
# This algorithm's performance drops significantly for frequencies higher than 0.25*RATE
# See report document for more explanation.


#************************************************** Parameters **************************************************

#-------------------------------YIN Algorithm Parameters------------------------------

max_T0 = 1/min_F0 #seconds
min_T0 = 1/max_F0 #seconds


max_tau_duration = max_T0 # maximum period seconds

# tau max is included
tau_max = int(np.round(RATE*max_tau_duration)) # one maximum period in samples

# window size must at least be greater than the maximum expected period.
# Include two maximum periods

window_len = 2 * tau_max #in samples
window_duration = window_len/RATE #seconds


Block_len = window_len + tau_max
Block_duration = Block_len/RATE # duration in seconds


#power of two above block_len
FFT_len = get_FFT_len(Block_len)


#------------------------------- Audio Load Parameters ------------------------------

#-------------------------------------------------------------
# 
#     DO NOT LEAVE WAV FILE EMPTY EVEN IF ONLY USING MIC
#
#    A DUMMY FILE IS NEEDED TO ENABLE KEYBOARD CHANGING OF INPUT MODE
#
#-------------------------------------------------------------

# wavefile = 'assets/piano_all_11025.wav'
# wavefile = 'assets/piano-G3.wav'
# wavefile = 'assets/trumpet_all.wav'
# wavefile = 'assets/violin-all.wav'
wavefile = 'assets/all_instruments.wav'


# Open wave file (should be mono channel)
wf = wave.open( wavefile, 'rb' )

# Read the wave file properties
num_channels    = wf.getnchannels()     # Number of channels
RATE_wave_file  = wf.getframerate()     # Sampling rate (frames/second)
signal_length   = wf.getnframes()       # Signal length
width           = wf.getsampwidth()     # Number of bytes per sample

#------------------------------- Setup PyAudio ------------------------------

# Open the audio output stream
p = pyaudio.PyAudio()

width_mic = 2
num_channels_mic = 1

PA_FORMAT = p.get_format_from_width(width_mic)
stream = p.open(
    format = PA_FORMAT,
    channels = num_channels_mic,
    rate = RATE,
    input = True,
    output = True,
    frames_per_buffer = Block_len)


#------------------------------- PyGame Parameters ------------------------------

fps = 60
WIDTH = 52 * 35
HEIGHT = 400

active_whites = []
active_blacks = []

#------------------------------- Piano Parameters ------------------------------

piano_freqs = np.array(pl.piano_freqs)
piano_notes = np.array(pl.piano_notes)

white_notes=pl.white_notes
black_notes=pl.black_notes

piano_taus = RATE/piano_freqs
piano_taus_integer = np.round(piano_taus)

duplicates = find_duplicates(piano_taus_integer)



#-------------------------------********------------------------------#-------------------------------********------------------------------
#-------------------------------********------------------------------#-------------------------------********------------------------------
#-------------------------------********------------------------------#-------------------------------********------------------------------




#************************************************** Start ***********************************************************************


print()
print('* Start *')
print()

#------------------------------- Print the Parameters ------------------------------

print("#------------          Recording File Parameters                 ----------#")
print('The audio file has %d channel(s).'            % num_channels)
print('The frame rate is %d frames/second.'    % RATE_wave_file)
print('The audio file has %d frames.'                % signal_length)
print('There are %d bytes per sample.'         % width)
print("#-------------------------------********------------------------------#")
print()

print("#------------         YIN Algorithm Parameters                 ----------#")
print('Sampling Rate in Hz: %.2f' % RATE)
print('Lowest detectable period in milliseconds: %.2f' % (1000.0 * min_T0))
print('Highest detectable period in milliseconds: %.2f' % (1000.0 * max_T0))
print()

print('Tau maximum (included) in samples : %d' % tau_max)
print('Maximum Tau duration in milliseconds: %.2f' % (1000*max_tau_duration))
print()

print('Window length in samples: %d' % window_len)
print('Duration of window in milliseconds: %.2f' % (1000*window_duration))
print()

print('Block length in samples: %d' % Block_len)
print('Duration of block in milliseconds: %.2f' % (1000*Block_duration))
print()

print('FFT length correspoinding to block length in samples: %d' % FFT_len)
print()

if FFT_len > 2048:
    warnings.warn("FFT length is larger then 2048. Consider lowering Sample Rate and/or increasing the minimum frequency", UserWarning)

print()
print("#-------------------------------********------------------------------#")



print("High Frequency Notes that CANNOT BE RESOLVED FROM EACH OTHER at Sampling Rate:", RATE)
print()

for dup in (duplicates[0]):
  dup_ind = np.where(piano_taus_integer == dup)[0]
 
  print(piano_notes[dup_ind])

print()

#------------------------------- Initialize PyGame ------------------------------

pygame.init()

font = pygame.font.Font('assets/Terserah.ttf', 48)
medium_font = pygame.font.Font('assets/Terserah.ttf', 28)
small_font = pygame.font.Font('assets/Terserah.ttf', 16)
real_small_font = pygame.font.Font('assets/Terserah.ttf', 10)


timer = pygame.time.Clock()
screen = pygame.display.set_mode([WIDTH, HEIGHT])
pygame.display.set_caption('YIN Pitch Detection')


#------------------------------- Algoritm Pitch Recordings ------------------------------

algo_running = False
mic_on = False
recording_on = False

fund_T0_recording = []
confidence_recording = []

Note_recording = []
Note_idx_recording = []

white_key_idx_recording = []
black_key_idx_recording = []

block_num = 0

#------------------------------- Run Program ------------------------------

run = True
while run:
  timer.tick(fps)
  screen.fill('gray')
  white_keys, black_keys, active_whites, active_blacks = draw_piano(active_whites, active_blacks)
  draw_title_bar()

  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      run = False


    if event.type == pygame.KEYDOWN:
      if event.key == pygame.K_SPACE and (not algo_running) and (mic_on or recording_on):
        print("Program Started!")
        print()
        algo_running = True

      # checking if key "M" was pressed
      if event.key == pygame.K_m and not mic_on:
          if not recording_on:
            print("Mic On!")
            print()
            mic_on = True

      # checking if key "R" was pressed
      elif event.key == pygame.K_r and not recording_on:
          if not mic_on:
            print("Recording On!")
            print()
            recording_on = True

            if RATE != RATE_wave_file:
              raise ValueError("The sampling rates of the audio and algorithm do not match. Program has stoped.")
          
     
        


  #------------------------------- Start the Algorithm ------------------------------
  if algo_running:
    # every frame, 16.66 ms, read a block

    if mic_on:
      # Read audio input stream
      input_bytes = stream.read(Block_len, exception_on_overflow = False)

      algo_width = width_mic

    elif recording_on:
      input_bytes = wf.readframes(Block_len)

      algo_width = width

    


    

    if len(input_bytes) >= algo_width * Block_len:
      
      # write to audio output stream
      if recording_on:
        stream.write(input_bytes, Block_len) 


      # Convert binary data to number sequence (numpy array)
      block_from_buffer = np.frombuffer(input_bytes, dtype = 'int16')

      # Convert to 16-bit integers by downcasting
      block_from_buffer = block_from_buffer.astype('int16')


      #normalize
      blockstd = np.std(block_from_buffer)
      if blockstd != 0:
        block_meanStd_norm = (block_from_buffer - np.mean(block_from_buffer))/blockstd
      else:
        block_meanStd_norm = (block_from_buffer - np.mean(block_from_buffer))
         

      try:
        fundamental_period_samples, confidence =\
        detect_pitch(block_meanStd_norm, window_len, tau_max,threshold)
      except Exception as e:
        print("An error occurred:", e)

      # get the closest note
      closest_note_index = np.argmin(np.abs(fundamental_period_samples - piano_taus))
      note = piano_notes[closest_note_index]

      if fundamental_period_samples >= tau_min and  confidence == threshold:
          
        if note in white_notes:
          white_key_idx = white_notes.index(note)
          white_key_idx_recording.append(white_key_idx)

          active_whites.append([white_key_idx, 1])

      
        else:
          black_key_idx = black_notes.index(note)
          black_key_idx_recording.append(black_key_idx)

          active_blacks.append([black_key_idx, 1])

       


      fund_T0_recording.append(fundamental_period_samples)
      confidence_recording.append(confidence)

      Note_recording.append(note)
      Note_idx_recording.append(closest_note_index)



      block_num += 1

      
                

  pygame.display.flip()



#************************************************** Finish **************************************************************************


stream.stop_stream()
stream.close()
p.terminate()

wf.close()

pygame.quit() 

print()
print('* Finished *')
print()