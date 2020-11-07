from aipollo_processor.detectors import half_note_detector
import django
from aipollo_processor import score_analyzer
import cv2


def index(request):
    template = django.template.loader.get_template('./scoreanalyzer/analyzer.html')

    # Debug: load sheet music scan from disk.
    image = cv2.imread('../sample_scans/bleib_rotated.jpg', cv2.IMREAD_GRAYSCALE)
    staffs, half_notes = score_analyzer.analyze_score(image)
    context = {
        'staffs': staffs,
        'half_notes': half_notes,
    }
    return django.http.HttpResponse(template.render(context, request))
    
    '''
    if request.method == 'POST':
        query = request.POST.get('query', None)

        # Get papers by similarity 
        result_list = [x[0] for x in similar_papers_service.get_papers_by_similarity(query)]
    else:
        result_list = [x[0] for x in similar_papers_service.get_papers_by_similarity('debug')]
    
    template = django.template.loader.get_template('searchinterface/index.html')
    context = {
        'result_list': result_list
    }
    return django.http.HttpResponse(template.render(context, request))
    '''