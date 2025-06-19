import datetime
import pedalboard as pb
import pandas as pd
import numpy as np
import os
import random
import soundfile as sf
import sounddevice as sd

from bayes_opt import BayesianOptimization
from bayes_opt import acquisition

import parselmouth

from psychopy import core, visual, event

stimulus_index = 0
win = None

## Path to the Indiana University Sentence Database
AUDIO_DIR = 'Indiana University Sentence Database/audio'
## List of the talkers 11 male and 10 female
SPEAKER = ["01","02","04","06","07","08","09","10","12","13","14","15","16","17","18","19","20","21","22","23","24"] 
EQ_BANDS_5 =  [200, 503, 1265, 3181, 8000]
Q = 2
RMS = True

S402_MIDDLE_ROW_AUSSEN = 'S402_middle_row_aussen.wav'
room = S402_MIDDLE_ROW_AUSSEN

def prepare_audio(talker_nb):
    '''
    Gets the talker and extract the audio files from the directory
    :param talker_nb: string
    :return: list of tuples (audio_file, audio_array, sample_rate)
    '''
    global audios
    audios = []
    audio_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith(f"{talker_nb}.wav")]
    for audio_file in audio_files:
        audio, sample_rate = get_sound_data(audio_file)
        audios.append((audio_file, audio, sample_rate))
    return audios

def get_sound_data(audio_file_name):
    '''
    Read the audio file and return the audio array and the sample rate
    gets exclusivly used by the prepare_audio function
    :param audio_file: string
    :return: numpy array, int
    '''
    sound_path = os.path.join(AUDIO_DIR, audio_file_name)
    audio, sample_rate = sf.read(sound_path)
    return audio, sample_rate

def preprocess_audio(audio, sample_rate):
    '''
    called right before playing the audio
    add a fade in and out to smooth the audio
    add the impulse response of the chosen roomreverberation
    does db correction if needed
    :param audio: numpy array
    :param sample_rate: int

    '''
    # Smooth fade in/out
    n_fade = 5000
    fade_in = np.linspace(0, 1, n_fade)**2  
    fade_out = np.linspace(1, 0, n_fade)**2
    n_buff = 2500
    padded_audio = np.concatenate([np.zeros(n_buff), audio, np.zeros(n_buff)])
    padded_audio[:n_fade] *= fade_in
    padded_audio[-n_fade:] *= fade_out
    audio = padded_audio
    
    # ROOM reverberance
    global IR 
    if IR != False:
        mixed = impulse_response(audio, sample_rate, IR)
        audio = mixed

    # db correction
    if RMS == True: 
        snd = parselmouth.Sound(audio)
        snd.scale_intensity(63)
        audio = snd.values[0]
    snd_intensity = np.round(parselmouth.Sound(audio).get_intensity(),1)
    print(snd_intensity, "db")
    return audio

def impulse_response(audio, sample_rate, room):
    '''
    adds the reveration of the chosen room to the audio
    gets exclusivly used by the preprocess_audio function 
    :param audio: numpy array
    :param sample_rate: int
    :param room: string
    :return: numpy array
    '''
    ir_audio, ir_sample_rate = sf.read(room, dtype='float32')
    pedalboard = pb.Pedalboard([
        pb.Convolution(ir_audio,sample_rate=ir_sample_rate )
            ])
    ir = pedalboard(audio, sample_rate)
    mixed = ir*1.5 + audio
    return mixed

def play_it(audio_array, sample_rate):
    '''
    Play the audio array using sounddevice library
    :param audio_array: numpy array
    :param sample_rate: int
    :return: None
    '''
    audio_array = preprocess_audio(audio_array,sample_rate)
    duration = len(audio_array) / sample_rate
    sd.play(audio_array, sample_rate)
    core.wait(duration+0.3)

def apply_bandpass_filter(audio_array, sample_rate, frequency, q, gain):
    '''
    Apply a bandpass filter to the audio array
    :param audio_array: numpy array
    :param sample_rate: int
    :param frequency: float
    :param q: float
    :param gain: float
    :return: numpy array
    '''
    pedalboard = pb.Pedalboard([
        pb.PeakFilter(cutoff_frequency_hz=frequency, gain_db=gain, q=q),
    ])
    return pedalboard(audio_array, sample_rate)

def apply_highpass_filter(audio_array, sample_rate, cutoff_frequency):
    '''
    Apply a highpass filter to the audio array
    :param audio_array: numpy array
    :param sample_rate: int
    :param cutoff_frequency: float
    :return: numpy array
    '''
    pedalboard = pb.Pedalboard([
        pb.HighpassFilter(cutoff_frequency_hz=cutoff_frequency),
    ])
    return pedalboard(audio_array, sample_rate)

def extract_pitch_mean_std(talker_nb):
    '''
    Extract the pitch of the audio files
    :param talker_nb: string
    :return: tuple (mean, std)
    '''
    audios = prepare_audio(talker_nb)
    all_f0_values = []
    for i in audios:
        sound = parselmouth.Sound(i[1])
        pitch = sound.to_pitch()
        f0_values = pitch.selected_array['frequency']
        f0_values[f0_values == 0] = np.nan
        f0_values = f0_values[~np.isnan(f0_values)]
        all_f0_values.extend(f0_values)
    f0 = (int(np.mean(all_f0_values)), int(np.std(all_f0_values))) 
    return f0

def present_trial_background():
    '''
    Present the background for the trial, Shows the F and J keys with a gray opacity, and the header text
    :return: None
    '''
    global win
    ## Trial Text Header
    text_header = visual.TextStim(win=win, name='text_header',
        text='What Version do you prefer?',
        font='Arial',
        pos=(0, 0.6), draggable=False, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    ## Trial Text F Stim gray
    text_F = visual.TextStim(win=win, name='text_F',
        text='F',
        font='Arial',
        pos=(-0.4, 0), draggable=False, height=0.2, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=0.5, 
        languageStyle='LTR',
        depth=-3.0);
    ## Trial Text J Stim gray
    text_J = visual.TextStim(win=win, name='text_J',
        text='J',
        font='Arial',
        pos=(0.4, 0), draggable=False, height=0.2, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=0.5, 
        languageStyle='LTR',
        depth=-4.0);
    
    text_header.draw()
    text_F.draw()
    text_J.draw()
    return

def confidence_quality_question(og_sound_first):
    '''
    Present the quality rating screen and get the user's input for how much they liked the manipulated version on a scale from 1 to 5
    :param og_sound_first: bool, True if the original sound was played first, False if the manipulated sound was played first
    :return: int [1, 5] as a rating of the manipulated sound
    '''
    global log_file_path
    manipulated_version = "J" if og_sound_first else "F"
    counted = "second (2.)" if og_sound_first else "first (1.)"
    question_text = visual.TextStim(win=win, name='Quality',
        text=f"How much do you like Version {manipulated_version} that was the {counted} Version\non a scale from 1 to 5?",
        font='Arial',
        pos=(0, 0.6), draggable=False, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0)
    
    question_text_2 = visual.TextStim(win=win, name='Auswahl',
        text='Select by pressing a number on the keyboard on a scale from \n1 = disliked it very much, \n3 = neutral/didnt liked it but didnt disliked it either, \n5 = liked it very much',
        font='Arial',
        height=0.05,
        pos=(0, 0.3), draggable=False, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0)
    answer_options = ['1', '2', '3', '4', '5', 'num_1', 'num_2', 'num_3', 'num_4', 'num_5']
    answer_texts = [visual.TextStim(win=win, text=f'{i}', height=0.3, opacity=0.5, font='Arial', pos=(x, -0.2), color='white') for i, x in zip(answer_options[:5], [-0.8, -0.4, 0, 0.4, 0.8])]
    for answer_text in answer_texts:
        answer_text.draw()
    question_text.draw()
    question_text_2.draw()
    win.flip()
    chosen_answer = None
    while chosen_answer not in answer_options:
        keys = event.waitKeys(keyList=answer_options)
        if keys:
            chosen_answer = keys[0]
    if len(chosen_answer) > 2:
        chosen_answer = chosen_answer[-1]
    x = [-0.8, -0.4, 0, 0.4, 0.8][answer_options.index(chosen_answer)]
    feedback_number = visual.TextStim(win=win, text=chosen_answer, height=0.3, font='Arial', pos=(x, -0.2), color='white')
    feedback_number.draw()
    win.flip()
    log_file = open(log_file_path, "a")
    log_file.write(f"Quality rating: {chosen_answer}\n")
    log_file.close()
    core.wait(0.1)
    return int(chosen_answer)

def confidence_rating():
    '''
    Present the confidence rating screen and get the user's input for the confidence of the preference on a scale from 1 to 9
    :return: int [1, 9] as a confidence rating of the preference
    '''
    global log_file_path
    question_text = visual.TextStim(win=win, name='Confidence',
        text='How sure are you about your choice on  a scale from 1 to 9?',
        font='Arial',
        pos=(0, 0.6), draggable=False, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0)
    
    question_text_2 = visual.TextStim(win=win, name='Auswahl',
        text='Select by pressing a number on the keyboard on a scale from \n1 = not confident at all, \n9 = very confident',
        font='Arial',
        height=0.05,
        pos=(0, 0.3), draggable=False, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0)
    answer_options = ['1', '2', '3','4','5','6','7','8','9','num_1', 'num_2', 'num_3','num_4','num_5','num_6','num_7','num_8','num_9']
    answer_texts = [visual.TextStim(win=win, text=f'{i}',height = 0.3,opacity=0.5,  font='Arial', pos=(x, -0.2), color='white') for i, x in zip(answer_options, [-0.8,-0.6,-0.4,-0.2, 0, 0.2,0.4,0.6,0.8])]
    for answer_text in answer_texts:
        answer_text.draw()
    question_text.draw()
    question_text_2.draw()
    win.flip()
    chosen_answer = None
    while chosen_answer not in answer_options:
        keys = event.waitKeys(keyList=answer_options)
        if keys:
            chosen_answer = keys[0]
    if len(chosen_answer)>2:
        chosen_answer = chosen_answer[-1] 
    x = [-0.8,-0.6,-0.4,-0.3, 0,0.3,0.4,0.6,0.8][answer_options.index(chosen_answer)]
    feedback_number = visual.TextStim(win=win, text=chosen_answer, height=0.3,  font='Arial', pos=(x, -0.2), color='white')
    feedback_number.draw()
    win.flip()
    log_file = open(log_file_path, "a")
    log_file.write(f"Confidence of: {chosen_answer}\n")
    log_file.close()
    core.wait(0.1)
    return int(chosen_answer)

def trial(audio_f, audio_j, sample_rate):
    '''
    Present the trial screen and get the user's input for the preference
    :param audio_f: numpy array
    :param audio_j: numpy array
    :param sample_rate: int
    :return: list of keys pressed 
    '''
    global timer
    if timer.getTime() > 360: ## timer to take a break every 6 Minutes
        break_text= visual.TextStim(win=win, text="Now its time for a short break. Please leave the chamber briefly.\n \nPress RETURN to continue with the next trial.")
        break_text.draw()
        win.flip()
        print("Time for a break")
        event.waitKeys(keyList=['return'])
        timer.reset()  # Setze den Timer zurück
    present_trial_background()
    win.flip()
    valid_keys = ['f', 'j', 'space','return']
    text_F_playing = visual.TextStim(win=win, name='text_F_playing',
        text='F',
        font='Arial',
        pos=(-0.4, 0), draggable=False, height=0.2, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=-6.0);
    
    text_J_playing = visual.TextStim(win=win, name='text_J_playing',
        text='J',
        font='Arial',
        pos=(0.4, 0), draggable=False, height=0.2, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=-10.0);
    
    instruction_text = visual.TextStim(win=win, name='instruction_text',
        text= 'Press...\n\"RETURN\" to listen again\n\"SPACE\" if you have no preference\n\"F\" if the first version is prefered\n\"J\" if the second version is prefered',
        font='Arial',
        pos=(0.2, -0.5), height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=-11.0, alignText='left')
    
    core.wait(0.1)
    present_trial_background()
    text_F_playing.draw()
    win.flip()
    play_it(audio_f, sample_rate)
    present_trial_background()
    win.flip()
    present_trial_background()
    text_J_playing.draw()
    win.flip()
    play_it(audio_j, sample_rate)
    win.flip()
    present_trial_background()
    instruction_text.draw()
    win.flip()
    log_file = open(log_file_path, "a")
    log_file.write(f"Start Time of the preference decision phase: {datetime.datetime.now()}\n")
    log_file.close()
    while True:
        keys = event.waitKeys(keyList=valid_keys)
        if "f" or "j" or "space" or "return" not in keys: ## This is a workaround for the bug in psychopy, that it does not recognize the keys correctly
            break
    win.flip()
    log_file = open(log_file_path, "a")
    log_file.write(f"End Time of the preference decision phase: {datetime.datetime.now()}\n")
    log_file.close()
    
    return keys    

def evaluate(sound_reference, audio, sample_rate):
    '''
    Evaluate the audio by comparing it to the reference sound
    it starts the trials in random order and handels the user input 
    or calls the confidence rating function if the user has no preference
    :param sound_reference: numpy array of the reference Version
    :param audio: numpy array of the manipulated sound
    :param sample_rate: int
    :return: int [-9, 9] as a evaluation of the manipulated sound
    '''
    global log_file_path
    inputted_keys = ['return']
    text_F_chosen = visual.TextStim(win=win, name='text_F_playing',
            text='F',
            font='Arial',
            pos=(-0.4, 0), draggable=False, height=0.2, wrapWidth=None, ori=0.0, 
            color='white', colorSpace='rgb', opacity=1.0, 
            languageStyle='LTR',
            depth=-6.0);
    text_J_chosen = visual.TextStim(win=win, name='text_J_playing',
        text='J',
        font='Arial',
        pos=(0.4, 0), draggable=False, height=0.2, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=1.0, 
        languageStyle='LTR',
        depth=-10.0);
    og_sound_first = random.choice([True, False])
    log_file = open(log_file_path, "a")   
    if og_sound_first:
        log_file.write(f"Played ORIGINAL Version first\n")
    else:
        log_file.write(f"Played MANIPULATED Version first\n")
    log_file.close()
    # log_file = open(log_file_path, "a")
    while inputted_keys[0] == 'return':
        if og_sound_first:
            inputted_keys = trial(sound_reference, audio, sample_rate)
        else:
            inputted_keys = trial(audio, sound_reference, sample_rate)
        if inputted_keys[0] =="return": 
            print("RETURN")
            log_file = open(log_file_path, "a")
            log_file.write(f"listend again\n")
            log_file.close()
            feedback_text = visual.TextStim(win=win, text="listen again")
            feedback_text.draw()
            win.flip()
            core.wait(0.1)
    if inputted_keys[0] == 'space':
        print("SPACE")
        log_file = open(log_file_path, "a")
        log_file.write("Preference for NONE\n")
        log_file.close()
        feedback_text = visual.TextStim(win=win, text="No Preference")
        feedback_text.draw()
        win.flip()
        core.wait(0.1)
        confidence_quality_question(og_sound_first)
        return 0 # does anyone have a better idea for this?
    elif inputted_keys[0] == 'f':
        print("F")
        text_F_chosen.draw()
        win.flip()
        core.wait(0.1)
        if og_sound_first:
            log_file = open(log_file_path, "a")
            log_file.write("Preference for ORIGINAL\n")
            log_file.close()
            return -1 * confidence_rating()
        else:
            log_file = open(log_file_path, "a")
            log_file.write("Preference for MANIPULATED\n")
            log_file.close()
            return 1 * confidence_rating()
    elif inputted_keys[0] == 'j':
        print("J")
        text_J_chosen.draw()
        win.flip()
        core.wait(0.1)    
        if og_sound_first:
            log_file = open(log_file_path, "a")
            log_file.write("Preference for MANIPULATED\n")
            log_file.close()
            return 1 * confidence_rating()
        else:
            log_file = open(log_file_path, "a")
            log_file.write("Preference for ORIGINAL\n")
            log_file.close()
            return -1 * confidence_rating()
    return 0

def optimize_cutoff_filter(cutoff_frequency):
    '''
    Optimize the cutoff frequency of the filter, 
    only does the highpass filtering
    calls the evaluate function to get the cost or win of the manipulated sound
    :param cutoff_frequency: float between 0 and 550
    :return: int [-9, 9] as a evaluation of the manipulated sound 
    '''
    global stimulus_index
    global audios
    global log_file_path
    global fundamental_mean_std
    cutoff_frequency = int(np.round(cutoff_frequency))
    log_file = open(log_file_path, "a")
    log_file.write(f"\nSentence: {stimulus_index}\n")
    log_file.write(f"Cutoff Frequency: {cutoff_frequency}\n")
    log_file.close()
    audio = audios[stimulus_index][1]
    sample_rate = audios[stimulus_index][2]
    reference_cutoff_point = 600 #Hz 
    sound_reference = apply_highpass_filter(audio, sample_rate, reference_cutoff_point) # this is the reference Version
    audio = apply_highpass_filter(audio, sample_rate, cutoff_frequency) # this is the manipulated Version
    
    cost_or_win = evaluate(sound_reference, audio, sample_rate)
    
    log_file = open(log_file_path, "a")
    log_file.write(f"Evaluation: {cost_or_win}\n")
    log_file.close()
    distance = abs(cutoff_frequency - fundamental_mean_std[0]) # the distance between the cutoff frequency and the f0 median
    log_file = open(log_file_path, "a")
    log_file.write(f"Distance from the pitch_mean: {distance}\n")
    log_file.write(f"reward: {cost_or_win}\n")
    log_file.close()
    return cost_or_win

def optimize_cutoff(n_trials):
    '''
    Optimize the cutoff frequency of the filter,
    only does the highpass filtering
    handle the baysian optimization 
    :param n_trials: int of the number of trials
    :return: string of the optimal cutoff frequency
    '''
    global stimulus_index
    global log_file_path
    log_file = open(log_file_path, "a")
    log_file.write(f"\nOptimizing Highpass Filter starting with Sentence {stimulus_index}\n")
    log_file.close()
    pbounds = {'cutoff_frequency': (0, 550)}
    optimizer = BayesianOptimization(
        f=optimize_cutoff_filter,
        acquisition_function= acquisition.UpperConfidenceBound(kappa=50),
        pbounds=pbounds
    )
    trial_nb = 1
    for i in range(0,n_trials):
        log_file = open(log_file_path, "a")
        log_file.write(f"\nTrial {trial_nb} of {n_trials}")
        log_file.close()
        trial_nb += 1
        next_point = optimizer.suggest()
        target = optimize_cutoff_filter(**next_point)
        optimizer.register(params=next_point, target=target)
        stimulus_index = (stimulus_index + 1) % len(audios)
        print("Aktuelles Target: ",target, " at ",(next_point))
        print("Optimal Highpass Cutoff Point until now: ",int(np.round(optimizer.max['params']['cutoff_frequency'])),"Hz")
        print("With ",optimizer.max['target'], " Points")

    result = int(np.round(optimizer.max['params']['cutoff_frequency']))
    response_string = f"Optimal Cutoff at {result}Hz"
    log_file = open(log_file_path, "a")
    log_file.write("\n"+response_string+"\n")
    return response_string 

def optimize_bandpass_filter(gain_at_200Hz, gain_at_503Hz, gain_at_1265Hz, gain_at_3181Hz, gain_at_8000Hz):
    '''
    Optimize the bandpass filter of the filter,
    only does the bandpass filtering
    calls the evaluate function to get the cost or win of given manipulated sound
    :param gain_at_200Hz: float between -18 and 12
    :param gain_at_503Hz: float between -18 and 12
    :param gain_at_1265Hz: float between -18 and 12
    :param gain_at_3181Hz: float between -18 and 12
    :param gain_at_8000Hz: float between -18 and 12
    :return: float
    '''
    global stimulus_index
    global audios
    global log_file_path
    bands = [int(np.round(gain_at_200Hz)), int(np.round(gain_at_503Hz)), int(np.round(gain_at_1265Hz)), int(np.round(gain_at_3181Hz)), int(np.round(gain_at_8000Hz))]
    sum_of_total_gain = sum(abs(band) for band in bands)

    log_file = open(log_file_path, "a")
    log_file.write(f"\nSentence: {stimulus_index}\n")
    log_file.write(f"Gain at 200Hz: {bands[0]}, Gain at 503Hz: {bands[1]}, Gain at 1265Hz: {bands[2]}, Gain at 3181Hz: {bands[3]}, Gain at 8000Hz: {bands[4]}\n")
    log_file.close()
    sound_reference = audios[stimulus_index][1]
    sample_rate = audios[stimulus_index][2]
    audio = sound_reference.copy()
    for i in range(0,len(bands)):
        audio = apply_bandpass_filter(audio, sample_rate, EQ_BANDS_5[i], Q, bands[i])
    cost_or_win = evaluate(sound_reference, audio, sample_rate)
    
    print("abs sum prior ", sum_of_total_gain)
    scaling_divisor = np.sum((np.array(bands) ** 2) / 18**2 )/len(bands)   
    print("this is the scaling factor for the sq prior",scaling_divisor)
    scaled_costs = cost_or_win / scaling_divisor

    log_file = open(log_file_path, "a")
    log_file.write(f"Evaluation: {cost_or_win}\n")
    log_file.write(f"Sum of the total applied gain: {sum_of_total_gain}\n")
    log_file.write(f"reward: {scaled_costs}\n")
    log_file.close()
    return scaled_costs

def optimize_band(n_trials):
    '''
    Optimize the bandpass filter of the filter,
    only does the bandpass filtering
    handle the baysian optimization
    :param n_trials: int of the number of trials
    :return: string of the optimal bandpass filter 
    '''
    global stimulus_index
    global log_file_path
    global win
    log_file = open(log_file_path, "a")
    log_file.write(f"\n\nOptimizing 5-Band Filter starting with Sentence {stimulus_index} \n")
    log_file.close()

    gain_boundaries = (-18, 12)
    pbounds = {'gain_at_200Hz': gain_boundaries, 'gain_at_503Hz': gain_boundaries, 'gain_at_1265Hz': gain_boundaries, 'gain_at_3181Hz': gain_boundaries, 'gain_at_8000Hz': gain_boundaries}
    optimizer = BayesianOptimization(
        f= optimize_bandpass_filter,
        acquisition_function= acquisition.UpperConfidenceBound(kappa=2.50),
        pbounds=pbounds
    )
    trial_nb = 1 
    for i in range(0,n_trials):
        log_file = open(log_file_path, "a")
        log_file.write(f"\nTrial {trial_nb} of {n_trials}")
        log_file.close()
        trial_nb += 1
        next_point = optimizer.suggest()
        target = optimize_bandpass_filter(**next_point)
        optimizer.register(params=next_point, target=target)
        stimulus_index = (stimulus_index + 1) % len(audios)
        
        print("Optimal Bandpass Filter until now:")
        max_params = optimizer.max['params']
        sorted_dict = dict(sorted(max_params.items(), key=lambda item: int(item[0].split("_")[-1][:-2])))
        sorted_dict = dict(sorted(max_params.items(), key=lambda item: int(item[0].split("_")[-1][:-2])))
        result_string = ""
        for i in sorted_dict.items():
            result_string += f"{str(i[0].split('_')[-1])}: {int(np.round(i[1]))}, " 
        print(result_string)
        print("Points of Optimal Bandpass Filter until now: ",np.round(optimizer.max['target'],3))

    max_params = optimizer.max['params']
    sorted_dict = dict(sorted(max_params.items(), key=lambda item: int(item[0].split("_")[-1][:-2])))
    log_file = open(log_file_path, "a")
    log_file.write("\nOptimal parameters: Gain at...\n")
    result_string = ""
    for i in sorted_dict.items():
        result_string += f"{str(i[0].split('_')[-1])}: {int(np.round(i[1]))}, " 
    log_file.write(result_string+ "\n")
    log_file.close()
    return result_string

def optimize(n_trials_cutoff, n_trials_band,rms_bandpass):
    '''
    Optimize the cutoff frequency and the bandpass filter of the filter,
    calls the optimize_cutoff and optimize_band functions
    :param n_trials_cutoff: int of the number of trials for the cutoff frequency
    :param n_trials_band: int of the number of trials for the bandpass filter
    :return: list of the optimal cutoff frequency and the optimal bandpass filter
    '''
    ### just for the demo purposes
    if n_trials_cutoff == 0:
        return ["No Highpass Filter", optimize_band(n_trials_band)]
    elif n_trials_band == 0:
        optimal_highpass_filter = optimize_cutoff(n_trials_cutoff)
        return [optimal_highpass_filter, "No Bandpass Filter"]
    ### just for the demo purposes
    global RMS
    RMS = False
    optimal_highpass_filter = optimize_cutoff(n_trials_cutoff)

    if rms_bandpass == True: RMS = True 
    optimal_bandpass_filter = optimize_band(n_trials_band)    
  
    return [optimal_highpass_filter, optimal_bandpass_filter]

def experiment(impulse_response, talker, n_trials_cutoff, n_trials_band, rms_bandpass):
    '''
    Main function to run the experiment, prepares the audio, sets filling up the log file with important infos and starts the optimization
    :param impulse_response: string of the path to the impulse response file
    :param talker: string of the talker number
    :param n_trials_cutoff: int of the number of trials for the cutoff frequency
    :param n_trials_band: int of the number of trials for the bandpass filter
    :param rms_bandpass: bool, True
    if the bandpass filter should be applied with RMS correction, False otherwise
    :return: list of the optimal cutoff frequency and the optimal bandpass filter
    '''
    #preparations
    global IR
    global fundamental_mean_std
    IR = impulse_response
    if(IR): print(IR) 
    else: print("No Room Reverberance added")
    fundamental_mean_std = extract_pitch_mean_std(talker)
    global audios
    
    log_file = open(log_file_path, "a")
    log_file.write(f"\n\nTalker: {talker}\n")
    log_file.write(f"Pitch Distribution: {fundamental_mean_std}\n")
    log_file.write(f"Room Impulse Response: {impulse_response}\n")
    log_file.write(f"This Block has {n_trials_cutoff+n_trials_band} Trials\n")
    log_file.write(f"RMS Bandpass: {rms_bandpass}")
    log_file.close()

    instruction_text = visual.TextStim(win=win, text=f"""Start of Block.
    
reminder: 
    Please focus on the overall listening comfort.
    If in one Version there is anything that bothers you, you probably prefer the other version.
    The versions you liked or disliked the most should receive the highest confidence scores.

Press RETURN to continue"""
                                       , color="white", height=0.075, wrapWidth = 1.5, alignHoriz='left', pos= (-0.75,0))
    instruction_text.draw()
    win.flip()
    event.waitKeys(keyList=['return'])
    win.flip()
    
    #Start this Block
    result = optimize(n_trials_cutoff, n_trials_band, rms_bandpass)
    return result

def testtrials(talker):
    '''
    Function to present the seven test trials, which are used to familiarize the user with the experiment
    :param talker: string of the talker number
    :return: None
    '''
    global stimulus_index
    global audios
    global win
    global log_file_path
    global IR 
    global timer 
    IR = room
    instruction_01 = visual.TextStim(win,text="""reminder from the printed instructions:
                                     
    In this experiment, you will hear two versions of the same sentence in every trial. 
                                     
    The display will highlight which one is currently being played. 
    
    Your task it to select your preferred version 
    by tapping the corresponding button on the keyboard. 

    press RETURN to Continue"""
                                    , color= "white",wrapWidth = 1.75, height= 0.07, alignHoriz='left', pos= (-0.85,0) )
    instruction_01.draw()
    win.flip()
    prepare_audio(talker)
    event.waitKeys(keyList=['return']) 
    instruction_02 = visual.TextStim(win,text="""\nHow to choose a preference? 
                                     
        Think of the lecture hall in which you are supposed to imagine yourself sitting. 
                                     
        Choose your preference depending on what you think which version:

            - is more pleasant to listen to in the long term,
                                     
            - you need to focus less on the audio to understand the meaning, 
                
            - that the voice recording sounds more natural to you 
            
            - or that it is more comfortable for you to listen to 
                
        press RETURN to Continue""", color= "white",wrapWidth = 1.75, height= 0.07, alignHoriz='left', pos= (-0.85,0) )
    instruction_02.draw() 
    win.flip()
    event.waitKeys(keyList=['return'])
    instruction_03 = visual.TextStim(win,text="""\n    Think about whether you like the version while listening to the stimuli. 
    So that you can make an intuitive quick decision.
                                     
                                        DO NOT OVERTHINK! 
                                     
    It may well happen that you hear no difference, 
    or hear a clear difference but have no preference, 
    in which case you are welcome to tap the SPACE bar, 
    which corresponds to a neutral evaluation.                                   

    press RETURN to Continue""", color= "white",wrapWidth = 1.75, height= 0.07, alignHoriz='left', pos= (-0.85,0) )
    instruction_03.draw() 
    win.flip()
    event.waitKeys(keyList=['return'])
    instruction_04 = visual.TextStim(win,text="""\n
After selecting your preference, you will be asked to indicate your certainty or conviction of your choice on a scale of 1 to 9, 
or to indicate your liking on one version on a scale on 1 to 5. 

Select your answer by pressing a number on the keyboard.

At regular intervals, you will be asked to take a break and then briefly leave the audio chamber. 

                                     press RETURN to Continue""", color= "white",wrapWidth = 1.75, height= 0.07, alignHoriz='left', pos= (-0.85,0) )
    instruction_04.draw() 
    win.flip()
    event.waitKeys(keyList=['return'])
   
    global audios
    win.flip()
    testtrialtext = visual.TextStim(win=win, text="We start with a few test trials. This is for you to learn how the experiment works\n\nPress RETURN to continue", color="white", height=0.1, pos=(0, 0))
    testtrialtext.draw()
    win.flip()
    event.waitKeys(keyList=['return'])

    present_trial_background()
    win.flip()
    log_file = open(log_file_path, "a")
    log_file.write(f"\n\nStarting Test Trials with Talker: {talker}\n")
    log_file.close()
    global RMS
    RMS = True
    list_of_test_trial_filters = [
        [apply_bandpass_filter, 8000, 0.1, 45],
        [apply_bandpass_filter,5000, 20, 25],
        [apply_highpass_filter, 8000],
        [apply_bandpass_filter, 50, 0.5,40],
        [apply_highpass_filter, 2000],
        [apply_highpass_filter, 0],
        [apply_bandpass_filter, 2000, 0.5, -25]]
    timer.reset()
    for i in range(0,len(list_of_test_trial_filters)):
        sample_rate = audios[stimulus_index][2]
        sound_reference = audios[stimulus_index][1]
        filter_function = list_of_test_trial_filters[i][0]
        filter_parameters = list_of_test_trial_filters[i][1:]
        if filter_function == apply_highpass_filter:
            audio = filter_function(sound_reference, sample_rate, *filter_parameters)
        elif filter_function == apply_bandpass_filter:
            audio = filter_function(sound_reference, sample_rate, *filter_parameters)
        else:
            print("Error: Filter function not recognized")
            return
        evaluate(sound_reference, audio, sample_rate)
        stimulus_index += 1
        
    text = visual.TextStim(win=win, text="Loading ...", color="white", height=0.1, pos=(0, 0))
    text.draw()
    win.flip()
    log_file = open(log_file_path, "a")
    log_file.write("finished test Trials\n")
    log_file.close()
    testtrialfinishedtext = visual.TextStim(win=win, text="The test trials are finished. Take a short break and leave the audio chamber briefly.\nAfter that break the data collection starts.\n\nPress RETURN to continue", color="white", height=0.1, pos=(0, 0))
    testtrialfinishedtext.draw()
    win.flip()
    event.waitKeys(keyList=['return'])
    return 

def main():
    '''
    Main function to run the experiment, prepares the audio, sets filling up the log file with important infos and starts the optimization
    :return: None
    '''
    global win
    global stimulus_index
    global audios
    global timer
    global log_file_path 
    timer = core.Clock()
    #Preparations
    participant_code = input("Please enter the participant code: ")
    participant_age = input("Please enter the participant age: ")
    participant_sex = input("Please enter the participant gender/sex(m/f/d): ")
    participant_hearing_self_assessment = input("Please enter the participant hearing self-assessment (1-5): ")	
    participant_hearing_loss = input("Please enter the participant Diagnosed Hearing Loss (None if nothing is known): ")
    print(f"Participant Code: {participant_code}, Age: {participant_age}, Sex: {participant_sex}")
    log_file_path = f"results/{participant_code}.txt"
    log_file = open(f"results/{participant_code}.txt", "w")
    log_file.write(f"Participant Code: {participant_code}\n")
    log_file.write(f"Participant Age: {participant_age}\n")
    log_file.write(f"Participant Sex: {participant_sex}\n")
    log_file.write(f"Participant Hearing Self Assessment: {participant_hearing_self_assessment}\n")
    log_file.write(f"Participant Diagnosed Hearing Loss: {participant_hearing_loss}\n")
    log_file.write(f"Experiment Date: {datetime.datetime.now()}\n")
    log_file.close()

    # Create a window and start with the demo experiment
    win = visual.Window([1680, 1050], units="norm", fullscr=False, screen=1, useFBO=False)  # Disable Framebuffer Object (FBO)
    win.flip()
  
    stimulus_index = 48
    talker = ["07","09"]
    testtrials("01")
    timer.reset()
    for i in talker:
        # Default room impulse response: room, n_trials_cutoff=15, n_trials_band=100, rms_bandpass=True
        result = experiment(impulse_response=room, talker=i, n_trials_cutoff=15, n_trials_band=100, rms_bandpass=True) 
                     
        log_file = open(log_file_path, "a")
        log_file.write(f"\nResults for this talker: \n{result}\n")
        log_file.close()
        if i != talker[-1]:
            break_text= visual.TextStim(win=win, text="End of this Block. Take a short break. Please leave the chamber briefly.\n \nPress RETURN to continue.")
            break_text.draw()
            win.flip()
            print("Time for a break due to the end of the block")
            event.waitKeys(keyList=['return'])
            timer.reset()  # reset the timer for the next block

    thank_you_text = visual.TextStim(win=win, text="Thank you for your participation!\nYou can leave the audio chamber\nPress RETURN to finish the experiment", color="white", height=0.1, pos=(0, 0))
    thank_you_text.draw()
    win.flip()
    log_file = open(log_file_path, "a")
    log_file.write(f"\nEnd of Experiment: {datetime.datetime.now()}")
    log_file.close()
    event.waitKeys(keyList=['return'])
    win.close()
    core.quit()

main()