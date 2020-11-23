import os
import django
import cv2
import jsonpickle

from aipollo_omr import score_analyzer


def index(request):
    template = django.template.loader.get_template(
        './scoreanalyzer/analyzer.html')

    # Debug: load sheet music scan from disk rather than getting them from the POST request.
    from django.contrib.staticfiles import finders
    path_to_image = finders.find('scoreanalyzer/bleib_rotated.jpg')
    image = cv2.imread(path_to_image, cv2.IMREAD_GRAYSCALE)

    # IF Debug
    path_to_json = finders.find('scoreanalyzer/debug_score_elements.json')
    with open(path_to_json, 'r') as f:
        score_elements_json = f.read()
    # ELSE
    #score_elements = score_analyzer.analyze_score(
    #    image, os.environ.get('AIPOLLO_MODELS_DIR'))
    #score_elements_json = jsonpickle.encode(score_elements, unpicklable=False)

    context = {
        'score_elements_json': score_elements_json,
    }
    return django.http.HttpResponse(template.render(context, request))