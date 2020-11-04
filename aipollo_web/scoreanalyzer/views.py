import django


def index(request):
    template = django.template.loader.get_template('scoreanalyzer/index.html')
    staffs = score_analyzer.analyze_score
    context = {
        'staffs': staffs
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