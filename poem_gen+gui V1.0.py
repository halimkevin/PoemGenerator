from tkinter import *
from tkinter.scrolledtext import ScrolledText
from poem_gen import*

def clickbutton():
    testTitle=inp.get()
    testTitle = np.array([testTitle])
    x_test_seq = x_tok.texts_to_sequences(testTitle)
    x_test = pad_sequences(x_test_seq, maxlen=MAXLEN_TITLE, padding='pre')  #Zero-padding di awal
    poem_result=decode_sequence(x_test[0].reshape(1,MAXLEN_TITLE))

    poemout.delete(1.0,END)
    poemout.insert(INSERT, poem_result)
    poemout.pack(fill=X)

bkcolor='#dedede'
font_type='Calibri'

root=Tk()
root.title('Generator Puisi Berbahasa Indonesia Menggunakan Kecerdasan Buatan')
root.geometry('600x600')
root.configure(background=bkcolor)

#title
guititle=Label(root, text="Generator Puisi", font=(font_type,15,'bold'),background=bkcolor,justify=LEFT)
guititle.pack(pady=(10,5))

frameTitle = Frame(background=bkcolor)
frameTitle.pack(fill=X,padx=10)

#content
subtitle = Label(frameTitle, text="Judul:",background=bkcolor, width=9)
subtitle.pack(side=LEFT)

genButton=Button(frameTitle,font=(font_type,12),text="Generate",command=clickbutton)
genButton.pack(side=RIGHT,padx=(10,0))

inp = Entry(frameTitle)
inp.pack(fill=X, expand=True)

frameContent = Frame(background=bkcolor)
frameContent.pack(fill=X,padx=10, pady=(5,10))

subcontent = Label(frameContent, text="Isi:",background=bkcolor, width=9)
subcontent.pack(side=LEFT,anchor=N)

poemout = ScrolledText(frameContent,background='#FFFFFF',relief=GROOVE,font=(font_type,12),)
poemout.pack(fill=X,expand=True)

#authors
author1 = "2301947281 - Felicia Limanta"
author2 = "2301866656 - Kevin Halim"

frameAuthor = Frame(background=bkcolor)
frameAuthor.pack(fill=X, padx=10, pady=(0,10))

createdBy = Label(frameAuthor, text="Dibuat oleh:",background=bkcolor, width=9)
createdBy.pack(side=LEFT, anchor=NW)

authors = Label(frameAuthor, text=author1+"\n"+author2, justify=LEFT, background=bkcolor)
authors.pack(side=LEFT)

root.mainloop()
