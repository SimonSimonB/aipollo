def find_staves_lines(score_image):
    row_mean = [0] * score_image.shape[0]
    for i in range(score_image.shape[0]):
        row = score_image[i, :]
        row_mean[i] = mean(row)
    #print(row_mean)
    #row_mean = np.average(score_image, axis=1)
    #print(row_mean)
  
    threshold = sorted(row_mean)[10] * 1.5
    print(f'Threshold: {threshold}')
    stavelines = []
    for i in range(1, score_image.shape[0]):
        if(row_mean[i] < threshold and not(row_mean[i-1] < threshold)):
            stavelines.append(i)
    
    return stavelines

def find_max_symmetric_staves_positions(score_image):
    stavelines = find_staves_lines(score_image)
    print(stavelines)
    if not(len(stavelines) % 5 == 0):
        raise Exception()
    
    staves = []
    for i in range(4, len(stavelines), 5):
        distance_above = stavelines[i-4] if i-4 == 0 else stavelines[i-4] - stavelines[i-5]
        distance_below = score_image.shape[0] - stavelines[i] if i == len(stavelines)-1 else stavelines[i+1] - stavelines[i]
        padding = (int)(min(distance_above, distance_below) / 2.0)
        staves.append((stavelines[i-4] - padding, stavelines[i] + padding))

    print(staves)
    return staves

def find_symmetric_staves_multiple(score_image, lines_around=3):
    stavelines = find_staves_lines(score_image)
    if not(len(stavelines) % 5 == 0):
        raise Exception()
    
    staves = []
    for i in range(4, len(stavelines), 5):
        padding = (int)(lines_around * mean([stavelines[i-3] - stavelines[i-4], stavelines[i-2] - stavelines[i-3], stavelines[i-1] - stavelines[i-2], stavelines[i] - stavelines[i-1]]))
        start = max(0, stavelines[i-4] - padding)
        end = min(score_image.shape[0] - 1, stavelines[i] + padding)
        staves.append((start, end))
    
    print(staves)
    return staves