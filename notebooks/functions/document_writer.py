import numpy as np
from docx import Document
from pandas import DataFrame

def docx_writer(num_people: int, data: DataFrame, path: str):

    chunks = np.array_split(data, num_people)
    count:int= 0
    
    for index, chunk in enumerate(chunks):
        doc = Document()
        for _ in range(len(chunk)):
            doc.add_paragraph("Post id: " + str(chunk['ids'][count]))
            doc.add_paragraph("Distance: " + str(chunk['distance'][count])+ "\n")

            try: 
                para = doc.add_paragraph(chunk['texts'][count])
                para.keep_together = True
                doc.add_page_break() 
            except:
                pass

            count += 1
            
        doc.save(f"{path}\file_{index}.docx")