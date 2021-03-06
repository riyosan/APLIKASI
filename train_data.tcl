#############################################################################
# Generated by PAGE version 4.19
#  in conjunction with Tcl version 8.6
#  May 03, 2019 03:39:11 AM +0700  platform: Windows NT
set vTcl(timestamp) ""


if {!$vTcl(borrow)} {

set vTcl(actual_gui_bg) #d9d9d9
set vTcl(actual_gui_fg) #000000
set vTcl(actual_gui_analog) #ececec
set vTcl(actual_gui_menu_analog) #ececec
set vTcl(actual_gui_menu_bg) #d9d9d9
set vTcl(actual_gui_menu_fg) #000000
set vTcl(complement_color) #d9d9d9
set vTcl(analog_color_p) #d9d9d9
set vTcl(analog_color_m) #ececec
set vTcl(active_fg) #000000
set vTcl(actual_gui_menu_active_bg)  #ececec
set vTcl(active_menu_fg) #000000
}

#############################################################################
# vTcl Code to Load User Fonts

vTcl:font:add_font \
    "-family {Segoe UI} -size 17 -weight bold -slant roman -underline 0 -overstrike 0" \
    user \
    vTcl:font10
vTcl:font:add_font \
    "-family {Segoe UI} -size 12 -weight bold -slant roman -underline 0 -overstrike 0" \
    user \
    vTcl:font9
#################################
#LIBRARY PROCEDURES
#


if {[info exists vTcl(sourcing)]} {

proc vTcl:project:info {} {
    set base .top42
    global vTcl
    set base $vTcl(btop)
    if {$base == ""} {
        set base .top42
    }
    namespace eval ::widgets::$base {
        set dflt,origin 0
        set runvisible 1
    }
    namespace eval ::widgets_bindings {
        set tagslist {_TopLevel _vTclBalloon}
    }
    namespace eval ::vTcl::modules::main {
        set procs {
        }
        set compounds {
        }
        set projectType single
    }
}
}

#################################
# GENERATED GUI PROCEDURES
#

proc vTclWindow.top42 {base} {
    if {$base == ""} {
        set base .top42
    }
    if {[winfo exists $base]} {
        wm deiconify $base; return
    }
    set top $base
    ###################
    # CREATING WIDGETS
    ###################
    vTcl::widgets::core::toplevel::createCmd $top -class Toplevel \
        -background {#d83838} -highlightbackground {#d9d9d9} \
        -highlightcolor black 
    wm focusmodel $top passive
    wm geometry $top 1366x705+-49+87
    update
    # set in toplevel.wgt.
    global vTcl
    global img_list
    set vTcl(save,dflt,origin) 0
    wm maxsize $top 1370 749
    wm minsize $top 120 1
    wm overrideredirect $top 0
    wm resizable $top 1 1
    wm deiconify $top
    wm title $top "Latih Data"
    vTcl:DefineAlias "$top" "trainToplevel" vTcl:Toplevel:WidgetProc "" 1
    canvas $top.can49 \
        -background {#d9d9d9} -borderwidth 2 -closeenough 1.0 -height 213 \
        -highlightbackground {#d9d9d9} -highlightcolor black \
        -insertbackground black -relief ridge -selectbackground {#c4c4c4} \
        -selectforeground black -width 243 
    vTcl:DefineAlias "$top.can49" "Canvas1" vTcl:WidgetProc "trainToplevel" 1
    set site_3_0 $top.can49
    labelframe $site_3_0.lab67 \
        -font TkDefaultFont -foreground black -text Labelframe \
        -background {#d9d9d9} -height 75 -highlightbackground {#d9d9d9} \
        -highlightcolor black -width 150 
    vTcl:DefineAlias "$site_3_0.lab67" "Labelframe1" vTcl:WidgetProc "trainToplevel" 1
    place $site_3_0.lab67 \
        -in $site_3_0 -x 500 -y 60 -anchor nw -bordermode ignore 
    canvas $top.can51 \
        -background {#d9d9d9} -borderwidth 2 -closeenough 1.0 -height 193 \
        -highlightbackground {#d9d9d9} -highlightcolor black \
        -insertbackground black -relief ridge -selectbackground {#c4c4c4} \
        -selectforeground black -width 243 
    vTcl:DefineAlias "$top.can51" "histEqCanvas1" vTcl:WidgetProc "trainToplevel" 1
    label $top.lab52 \
        -activebackground {#f9f9f9} -activeforeground black \
        -background {#d83838} -disabledforeground {#a3a3a3} \
        -font $::vTcl(fonts,vTcl:font9,object) -foreground {#000000} \
        -highlightbackground {#d9d9d9} -highlightcolor black \
        -text {Gambar Awal} 
    vTcl:DefineAlias "$top.lab52" "Label1" vTcl:WidgetProc "trainToplevel" 1
    label $top.lab53 \
        -activebackground {#f9f9f9} -activeforeground black \
        -background {#d83838} -disabledforeground {#a3a3a3} \
        -font $::vTcl(fonts,vTcl:font9,object) -foreground {#000000} \
        -highlightbackground {#d9d9d9} -highlightcolor black \
        -text {Histogram Equalization} 
    vTcl:DefineAlias "$top.lab53" "Label1_5" vTcl:WidgetProc "trainToplevel" 1
    label $top.lab54 \
        -activebackground {#f9f9f9} -activeforeground black \
        -background {#d83838} -disabledforeground {#a3a3a3} \
        -font $::vTcl(fonts,vTcl:font10,object) -foreground {#000000} \
        -highlightbackground {#d9d9d9} -highlightcolor black \
        -text {LATIH DATA} -width 200 
    vTcl:DefineAlias "$top.lab54" "Label1_6" vTcl:WidgetProc "trainToplevel" 1
    text $top.tex55 \
        -background white -font font14 -foreground black \
        -highlightbackground {#d9d9d9} -highlightcolor black \
        -insertbackground black -selectbackground {#c4c4c4} \
        -selectforeground black -width 224 -wrap word 
    .top42.tex55 configure -font font14
    .top42.tex55 insert end text
    vTcl:DefineAlias "$top.tex55" "imgFolderText" vTcl:WidgetProc "trainToplevel" 1
    label $top.lab56 \
        -activebackground {#f9f9f9} -activeforeground black \
        -background {#d83838} -disabledforeground {#a3a3a3} \
        -font $::vTcl(fonts,vTcl:font9,object) -foreground {#000000} \
        -highlightbackground {#d9d9d9} -highlightcolor black \
        -text {Pilih Folder Gambar} 
    vTcl:DefineAlias "$top.lab56" "Label1_7" vTcl:WidgetProc "trainToplevel" 1
    button $top.but57 \
        -activebackground {#ececec} -activeforeground {#000000} \
        -background {#d9d9d9} -disabledforeground {#a3a3a3} \
        -font TkDefaultFont -foreground {#000000} \
        -highlightbackground {#d9d9d9} -highlightcolor black -pady 0 \
        -text Browse 
    vTcl:DefineAlias "$top.but57" "browseButton1" vTcl:WidgetProc "trainToplevel" 1
    label $top.lab59 \
        -activebackground {#f9f9f9} -activeforeground black \
        -background {#d83838} -disabledforeground {#a3a3a3} \
        -font $::vTcl(fonts,vTcl:font9,object) -foreground {#000000} \
        -highlightbackground {#d9d9d9} -highlightcolor black \
        -text {Lihat Gambar} 
    vTcl:DefineAlias "$top.lab59" "Label1_8" vTcl:WidgetProc "trainToplevel" 1
    button $top.but60 \
        -activebackground {#ececec} -activeforeground {#000000} \
        -background {#d9d9d9} -disabledforeground {#a3a3a3} \
        -font TkDefaultFont -foreground {#000000} \
        -highlightbackground {#d9d9d9} -highlightcolor black -pady 0 \
        -text Miring 
    vTcl:DefineAlias "$top.but60" "miringImgButton" vTcl:WidgetProc "trainToplevel" 1
    button $top.but61 \
        -activebackground {#ececec} -activeforeground {#000000} \
        -background {#d9d9d9} -disabledforeground {#a3a3a3} \
        -font TkDefaultFont -foreground {#000000} \
        -highlightbackground {#d9d9d9} -highlightcolor black -pady 0 \
        -text Tegak 
    vTcl:DefineAlias "$top.but61" "tegakImgButton" vTcl:WidgetProc "trainToplevel" 1
    button $top.but62 \
        -activebackground {#ececec} -activeforeground {#000000} \
        -background {#d9d9d9} -disabledforeground {#a3a3a3} \
        -font TkDefaultFont -foreground {#000000} \
        -highlightbackground {#d9d9d9} -highlightcolor black -pady 0 \
        -text Spasi 
    vTcl:DefineAlias "$top.but62" "spasiImgButton" vTcl:WidgetProc "trainToplevel" 1
    button $top.but63 \
        -activebackground {#ececec} -activeforeground {#000000} \
        -background {#d9d9d9} -disabledforeground {#a3a3a3} \
        -font TkDefaultFont -foreground {#000000} \
        -highlightbackground {#d9d9d9} -highlightcolor black -pady 0 -text << 
    vTcl:DefineAlias "$top.but63" "geserKiriImgButton" vTcl:WidgetProc "trainToplevel" 1
    button $top.but64 \
        -activebackground {#ececec} -activeforeground {#000000} \
        -background {#d9d9d9} -disabledforeground {#a3a3a3} \
        -font TkDefaultFont -foreground {#000000} \
        -highlightbackground {#d9d9d9} -highlightcolor black -pady 0 -text >> 
    vTcl:DefineAlias "$top.but64" "geserKananImgButton" vTcl:WidgetProc "trainToplevel" 1
    button $top.but68 \
        -activebackground {#ececec} -activeforeground {#000000} \
        -background {#d9d9d9} -disabledforeground {#a3a3a3} \
        -font TkDefaultFont -foreground {#000000} \
        -highlightbackground {#d9d9d9} -highlightcolor black -pady 0 \
        -text {Latih Data} 
    vTcl:DefineAlias "$top.but68" "browseButton1_11" vTcl:WidgetProc "trainToplevel" 1
    canvas $top.can70 \
        -background {#d9d9d9} -borderwidth 2 -closeenough 1.0 -height 213 \
        -highlightbackground {#d9d9d9} -highlightcolor black \
        -insertbackground black -relief ridge -selectbackground {#c4c4c4} \
        -selectforeground black -width 243 
    vTcl:DefineAlias "$top.can70" "confMatMiringCanvas" vTcl:WidgetProc "trainToplevel" 1
    set site_3_0 $top.can70
    labelframe $site_3_0.lab67 \
        -font TkDefaultFont -foreground black -text Labelframe \
        -background {#d9d9d9} -height 75 -highlightbackground {#d9d9d9} \
        -highlightcolor black -width 150 
    vTcl:DefineAlias "$site_3_0.lab67" "Labelframe1_10" vTcl:WidgetProc "trainToplevel" 1
    place $site_3_0.lab67 \
        -in $site_3_0 -x 500 -y 60 -anchor nw -bordermode ignore 
    canvas $top.can71 \
        -background {#d9d9d9} -borderwidth 2 -closeenough 1.0 -height 213 \
        -highlightbackground {#d9d9d9} -highlightcolor black \
        -insertbackground black -relief ridge -selectbackground {#c4c4c4} \
        -selectforeground black -width 243 
    vTcl:DefineAlias "$top.can71" "confMatTegakCanvas" vTcl:WidgetProc "trainToplevel" 1
    set site_3_0 $top.can71
    labelframe $site_3_0.lab67 \
        -font TkDefaultFont -foreground black -text Labelframe \
        -background {#d9d9d9} -height 75 -highlightbackground {#d9d9d9} \
        -highlightcolor black -width 150 
    vTcl:DefineAlias "$site_3_0.lab67" "Labelframe1_12" vTcl:WidgetProc "trainToplevel" 1
    place $site_3_0.lab67 \
        -in $site_3_0 -x 500 -y 60 -anchor nw -bordermode ignore 
    canvas $top.can72 \
        -background {#d9d9d9} -borderwidth 2 -closeenough 1.0 -height 213 \
        -highlightbackground {#d9d9d9} -highlightcolor black \
        -insertbackground black -relief ridge -selectbackground {#c4c4c4} \
        -selectforeground black -width 243 
    vTcl:DefineAlias "$top.can72" "confMatSpasiCanvas" vTcl:WidgetProc "trainToplevel" 1
    set site_3_0 $top.can72
    labelframe $site_3_0.lab67 \
        -font TkDefaultFont -foreground black -text Labelframe \
        -background {#d9d9d9} -height 75 -highlightbackground {#d9d9d9} \
        -highlightcolor black -width 150 
    vTcl:DefineAlias "$site_3_0.lab67" "Labelframe1_14" vTcl:WidgetProc "trainToplevel" 1
    place $site_3_0.lab67 \
        -in $site_3_0 -x 500 -y 60 -anchor nw -bordermode ignore 
    label $top.lab73 \
        -activebackground {#f9f9f9} -activeforeground black \
        -background {#d83838} -disabledforeground {#a3a3a3} \
        -font $::vTcl(fonts,vTcl:font9,object) -foreground {#000000} \
        -highlightbackground {#d9d9d9} -highlightcolor black \
        -text {Confusion Matrix} 
    vTcl:DefineAlias "$top.lab73" "Label1_9" vTcl:WidgetProc "trainToplevel" 1
    label $top.lab74 \
        -activebackground {#f9f9f9} -activeforeground black \
        -background {#d83838} -disabledforeground {#a3a3a3} \
        -font $::vTcl(fonts,vTcl:font9,object) -foreground {#000000} \
        -highlightbackground {#d9d9d9} -highlightcolor black -text Kemiringan 
    vTcl:DefineAlias "$top.lab74" "Label1_10" vTcl:WidgetProc "trainToplevel" 1
    label $top.lab75 \
        -activebackground {#f9f9f9} -activeforeground black \
        -background {#d83838} -disabledforeground {#a3a3a3} \
        -font $::vTcl(fonts,vTcl:font9,object) -foreground {#000000} \
        -highlightbackground {#d9d9d9} -highlightcolor black -text Tegak 
    vTcl:DefineAlias "$top.lab75" "Label1_1" vTcl:WidgetProc "trainToplevel" 1
    label $top.lab76 \
        -activebackground {#f9f9f9} -activeforeground black \
        -background {#d83838} -disabledforeground {#a3a3a3} \
        -font $::vTcl(fonts,vTcl:font9,object) -foreground {#000000} \
        -highlightbackground {#d9d9d9} -highlightcolor black -text Spasi 
    vTcl:DefineAlias "$top.lab76" "Label1_2" vTcl:WidgetProc "trainToplevel" 1
    label $top.lab77 \
        -activebackground {#f9f9f9} -activeforeground black \
        -background {#d83838} -disabledforeground {#a3a3a3} \
        -font $::vTcl(fonts,vTcl:font9,object) -foreground {#000000} \
        -highlightbackground {#d9d9d9} -highlightcolor black \
        -text {Hasil Pelatihan Data} 
    vTcl:DefineAlias "$top.lab77" "Label1_3" vTcl:WidgetProc "trainToplevel" 1
    label $top.lab78 \
        -activebackground {#f9f9f9} -activeforeground black \
        -background {#d83838} -disabledforeground {#a3a3a3} \
        -font $::vTcl(fonts,vTcl:font9,object) -foreground {#000000} \
        -highlightbackground {#d9d9d9} -highlightcolor black \
        -text {Confusion Matrix Summary} 
    vTcl:DefineAlias "$top.lab78" "Label1_4" vTcl:WidgetProc "trainToplevel" 1
    vTcl::widgets::ttk::scrolledtext::CreateCmd $top.scr79 \
        -background {#d9d9d9} -height 75 -highlightbackground {#d9d9d9} \
        -highlightcolor black -width 125 
    vTcl:DefineAlias "$top.scr79" "confMatMiringScrolledtext" vTcl:WidgetProc "trainToplevel" 1

    $top.scr79.01 configure -background white \
        -font TkTextFont \
        -foreground black \
        -height 3 \
        -highlightbackground #d9d9d9 \
        -highlightcolor black \
        -insertbackground black \
        -insertborderwidth 3 \
        -selectbackground #c4c4c4 \
        -selectforeground black \
        -width 10 \
        -wrap none
    vTcl::widgets::ttk::scrolledtext::CreateCmd $top.scr81 \
        -background {#d9d9d9} -height 75 -highlightbackground {#d9d9d9} \
        -highlightcolor black -width 125 
    vTcl:DefineAlias "$top.scr81" "Scrolledtext2" vTcl:WidgetProc "trainToplevel" 1

    $top.scr81.01 configure -background white \
        -font TkTextFont \
        -foreground black \
        -height 3 \
        -highlightbackground #d9d9d9 \
        -highlightcolor black \
        -insertbackground black \
        -insertborderwidth 3 \
        -selectbackground #c4c4c4 \
        -selectforeground black \
        -width 10 \
        -wrap none
    vTcl::widgets::ttk::scrolledtext::CreateCmd $top.scr82 \
        -background {#d9d9d9} -height 75 -highlightbackground {#d9d9d9} \
        -highlightcolor black -width 125 
    vTcl:DefineAlias "$top.scr82" "Scrolledtext3" vTcl:WidgetProc "trainToplevel" 1

    $top.scr82.01 configure -background white \
        -font TkTextFont \
        -foreground black \
        -height 3 \
        -highlightbackground #d9d9d9 \
        -highlightcolor black \
        -insertbackground black \
        -insertborderwidth 3 \
        -selectbackground #c4c4c4 \
        -selectforeground black \
        -width 10 \
        -wrap none
    label $top.lab83 \
        -activebackground {#f9f9f9} -activeforeground black \
        -background {#d83838} -disabledforeground {#a3a3a3} \
        -font $::vTcl(fonts,vTcl:font9,object) -foreground {#000000} \
        -highlightbackground {#d9d9d9} -highlightcolor black -text Kemiringan 
    vTcl:DefineAlias "$top.lab83" "Label1_11" vTcl:WidgetProc "trainToplevel" 1
    label $top.lab84 \
        -activebackground {#f9f9f9} -activeforeground black \
        -background {#d83838} -disabledforeground {#a3a3a3} \
        -font $::vTcl(fonts,vTcl:font9,object) -foreground {#000000} \
        -highlightbackground {#d9d9d9} -highlightcolor black -text Tegak 
    vTcl:DefineAlias "$top.lab84" "Label1_2" vTcl:WidgetProc "trainToplevel" 1
    label $top.lab85 \
        -activebackground {#f9f9f9} -activeforeground black \
        -background {#d83838} -disabledforeground {#a3a3a3} \
        -font $::vTcl(fonts,vTcl:font9,object) -foreground {#000000} \
        -highlightbackground {#d9d9d9} -highlightcolor black -text Spasi 
    vTcl:DefineAlias "$top.lab85" "Label1_3" vTcl:WidgetProc "trainToplevel" 1
    text $top.tex86 \
        -background white -font TkTextFont -foreground black \
        -highlightbackground {#d9d9d9} -highlightcolor black \
        -insertbackground black -selectbackground {#c4c4c4} \
        -selectforeground black -width 64 -wrap word 
    .top42.tex86 configure -font TkTextFont
    .top42.tex86 insert end text
    vTcl:DefineAlias "$top.tex86" "nInputText" vTcl:WidgetProc "trainToplevel" 1
    label $top.lab87 \
        -activebackground {#f9f9f9} -activeforeground black \
        -background {#d83838} -disabledforeground {#a3a3a3} \
        -font $::vTcl(fonts,vTcl:font9,object) -foreground {#000000} \
        -highlightbackground {#d9d9d9} -highlightcolor black \
        -text {Jumlah Unit Input} 
    vTcl:DefineAlias "$top.lab87" "Label1_4" vTcl:WidgetProc "trainToplevel" 1
    label $top.lab89 \
        -activebackground {#f9f9f9} -activeforeground black \
        -background {#d83838} -disabledforeground {#a3a3a3} \
        -font $::vTcl(fonts,vTcl:font9,object) -foreground {#000000} \
        -highlightbackground {#d9d9d9} -highlightcolor black \
        -text {Pengaturan Parameter LVQ} 
    vTcl:DefineAlias "$top.lab89" "Label1_12" vTcl:WidgetProc "trainToplevel" 1
    label $top.lab90 \
        -activebackground {#f9f9f9} -activeforeground black \
        -background {#d83838} -disabledforeground {#a3a3a3} \
        -font $::vTcl(fonts,vTcl:font9,object) -foreground {#000000} \
        -highlightbackground {#d9d9d9} -highlightcolor black -text Epoch 
    vTcl:DefineAlias "$top.lab90" "Label1_5" vTcl:WidgetProc "trainToplevel" 1
    label $top.lab92 \
        -activebackground {#f9f9f9} -activeforeground black \
        -background {#d83838} -disabledforeground {#a3a3a3} \
        -font $::vTcl(fonts,vTcl:font9,object) -foreground {#000000} \
        -highlightbackground {#d9d9d9} -highlightcolor black \
        -text {Learning Rate} 
    vTcl:DefineAlias "$top.lab92" "Label1_13" vTcl:WidgetProc "trainToplevel" 1
    text $top.tex93 \
        -background white -font TkTextFont -foreground black \
        -highlightbackground {#d9d9d9} -highlightcolor black \
        -insertbackground black -selectbackground {#c4c4c4} \
        -selectforeground black -width 64 -wrap word 
    .top42.tex93 configure -font TkTextFont
    .top42.tex93 insert end text
    vTcl:DefineAlias "$top.tex93" "epochText" vTcl:WidgetProc "trainToplevel" 1
    text $top.tex94 \
        -background white -font TkTextFont -foreground black \
        -highlightbackground {#d9d9d9} -highlightcolor black \
        -insertbackground black -selectbackground {#c4c4c4} \
        -selectforeground black -width 64 -wrap word 
    .top42.tex94 configure -font TkTextFont
    .top42.tex94 insert end text
    vTcl:DefineAlias "$top.tex94" "learnRateText" vTcl:WidgetProc "trainToplevel" 1
    text $top.tex95 \
        -background white -font font14 -foreground black \
        -highlightbackground {#d9d9d9} -highlightcolor black \
        -insertbackground black -selectbackground {#c4c4c4} \
        -selectforeground black -width 224 -wrap word 
    .top42.tex95 configure -font font14
    .top42.tex95 insert end text
    vTcl:DefineAlias "$top.tex95" "saveModelText" vTcl:WidgetProc "trainToplevel" 1
    button $top.but96 \
        -activebackground {#ececec} -activeforeground {#000000} \
        -background {#d9d9d9} -disabledforeground {#a3a3a3} \
        -font TkDefaultFont -foreground {#000000} \
        -highlightbackground {#d9d9d9} -highlightcolor black -pady 0 \
        -text Browse 
    vTcl:DefineAlias "$top.but96" "saveBrowseButton" vTcl:WidgetProc "trainToplevel" 1
    label $top.lab97 \
        -activebackground {#f9f9f9} -activeforeground black \
        -background {#d83838} -disabledforeground {#a3a3a3} \
        -font $::vTcl(fonts,vTcl:font9,object) -foreground {#000000} \
        -highlightbackground {#d9d9d9} -highlightcolor black \
        -text {Simpan Model} 
    vTcl:DefineAlias "$top.lab97" "Label1_6" vTcl:WidgetProc "trainToplevel" 1
    button $top.but43 \
        -activebackground {#ececec} -activeforeground {#000000} \
        -background {#d9d9d9} -disabledforeground {#a3a3a3} \
        -font TkDefaultFont -foreground {#000000} \
        -highlightbackground {#d9d9d9} -highlightcolor black -pady 0 \
        -text {Halaman Awal} 
    vTcl:DefineAlias "$top.but43" "homeButton" vTcl:WidgetProc "trainToplevel" 1
    button $top.but44 \
        -activebackground {#ececec} -activeforeground {#000000} \
        -background {#d9d9d9} -disabledforeground {#a3a3a3} \
        -font TkDefaultFont -foreground {#000000} \
        -highlightbackground {#d9d9d9} -highlightcolor black -pady 0 \
        -text Keluar 
    vTcl:DefineAlias "$top.but44" "exitButton" vTcl:WidgetProc "trainToplevel" 1
    ###################
    # SETTING GEOMETRY
    ###################
    place $top.can49 \
        -in $top -x 30 -y 250 -width 243 -relwidth 0 -height 213 -relheight 0 \
        -anchor nw -bordermode ignore 
    place $top.can51 \
        -in $top -x 30 -y 510 -width 243 -relwidth 0 -height 193 -relheight 0 \
        -anchor nw -bordermode ignore 
    place $top.lab52 \
        -in $top -x 30 -y 220 -width 104 -relwidth 0 -height 21 -relheight 0 \
        -anchor nw -bordermode ignore 
    place $top.lab53 \
        -in $top -x 30 -y 480 -width 184 -relwidth 0 -height 21 -relheight 0 \
        -anchor nw -bordermode ignore 
    place $top.lab54 \
        -in $top -x 600 -y 10 -width 164 -relwidth 0 -height 21 -relheight 0 \
        -anchor nw -bordermode ignore 
    place $top.tex55 \
        -in $top -x 30 -y 80 -width 224 -relwidth 0 -height 24 -relheight 0 \
        -anchor nw -bordermode ignore 
    place $top.lab56 \
        -in $top -x 20 -y 50 -width 174 -relwidth 0 -height 21 -relheight 0 \
        -anchor nw -bordermode ignore 
    place $top.but57 \
        -in $top -x 260 -y 80 -width 87 -relwidth 0 -height 24 -relheight 0 \
        -anchor nw -bordermode ignore 
    place $top.lab59 \
        -in $top -x 30 -y 130 -width 104 -relwidth 0 -height 21 -relheight 0 \
        -anchor nw -bordermode ignore 
    place $top.but60 \
        -in $top -x 30 -y 160 -width 87 -relwidth 0 -height 24 -relheight 0 \
        -anchor nw -bordermode ignore 
    place $top.but61 \
        -in $top -x 130 -y 160 -width 87 -height 24 -anchor nw \
        -bordermode ignore 
    place $top.but62 \
        -in $top -x 230 -y 160 -width 87 -height 24 -anchor nw \
        -bordermode ignore 
    place $top.but63 \
        -in $top -x 110 -y 190 -width 27 -relwidth 0 -height 24 -relheight 0 \
        -anchor nw -bordermode ignore 
    place $top.but64 \
        -in $top -x 210 -y 190 -width 27 -height 24 -anchor nw \
        -bordermode ignore 
    place $top.but68 \
        -in $top -x 940 -y 120 -width 87 -height 24 -anchor nw \
        -bordermode ignore 
    place $top.can70 \
        -in $top -x 440 -y 240 -width 243 -height 213 -anchor nw \
        -bordermode ignore 
    place $top.can71 \
        -in $top -x 700 -y 240 -width 243 -height 213 -anchor nw \
        -bordermode ignore 
    place $top.can72 \
        -in $top -x 960 -y 240 -width 243 -height 213 -anchor nw \
        -bordermode ignore 
    place $top.lab73 \
        -in $top -x 439 -y 190 -width 134 -relwidth 0 -height 21 -relheight 0 \
        -anchor nw -bordermode ignore 
    place $top.lab74 \
        -in $top -x 438 -y 210 -width 94 -relwidth 0 -height 21 -relheight 0 \
        -anchor nw -bordermode ignore 
    place $top.lab75 \
        -in $top -x 700 -y 210 -width 44 -relwidth 0 -height 21 -relheight 0 \
        -anchor nw -bordermode ignore 
    place $top.lab76 \
        -in $top -x 960 -y 210 -width 44 -relwidth 0 -height 21 -relheight 0 \
        -anchor nw -bordermode ignore 
    place $top.lab77 \
        -in $top -x 440 -y 170 -width 154 -relwidth 0 -height 21 -relheight 0 \
        -anchor nw -bordermode ignore 
    place $top.lab78 \
        -in $top -x 438 -y 460 -width 214 -relwidth 0 -height 21 -relheight 0 \
        -anchor nw -bordermode ignore 
    place $top.scr79 \
        -in $top -x 440 -y 510 -width 243 -relwidth 0 -height 161 \
        -relheight 0 -anchor nw -bordermode ignore 
    place $top.scr81 \
        -in $top -x 700 -y 510 -width 243 -relwidth 0 -height 161 \
        -relheight 0 -anchor nw -bordermode ignore 
    place $top.scr82 \
        -in $top -x 960 -y 510 -width 243 -relwidth 0 -height 161 \
        -relheight 0 -anchor nw -bordermode ignore 
    place $top.lab83 \
        -in $top -x 438 -y 480 -width 94 -height 21 -anchor nw \
        -bordermode ignore 
    place $top.lab84 \
        -in $top -x 700 -y 480 -width 44 -height 21 -anchor nw \
        -bordermode ignore 
    place $top.lab85 \
        -in $top -x 960 -y 480 -width 44 -height 21 -anchor nw \
        -bordermode ignore 
    place $top.tex86 \
        -in $top -x 600 -y 80 -width 64 -relwidth 0 -height 24 -relheight 0 \
        -anchor nw -bordermode ignore 
    place $top.lab87 \
        -in $top -x 440 -y 80 -width 144 -relwidth 0 -height 21 -relheight 0 \
        -anchor nw -bordermode ignore 
    place $top.lab89 \
        -in $top -x 440 -y 50 -width 214 -relwidth 0 -height 21 -relheight 0 \
        -anchor nw -bordermode ignore 
    place $top.lab90 \
        -in $top -x 690 -y 80 -width 54 -relwidth 0 -height 21 -relheight 0 \
        -anchor nw -bordermode ignore 
    place $top.lab92 \
        -in $top -x 840 -y 80 -width 114 -relwidth 0 -height 21 -relheight 0 \
        -anchor nw -bordermode ignore 
    place $top.tex93 \
        -in $top -x 750 -y 80 -width 64 -height 24 -anchor nw \
        -bordermode ignore 
    place $top.tex94 \
        -in $top -x 960 -y 80 -width 64 -height 24 -anchor nw \
        -bordermode ignore 
    place $top.tex95 \
        -in $top -x 600 -y 120 -width 224 -height 24 -anchor nw \
        -bordermode ignore 
    place $top.but96 \
        -in $top -x 830 -y 120 -width 87 -height 24 -anchor nw \
        -bordermode ignore 
    place $top.lab97 \
        -in $top -x 449 -y 120 -width 114 -relwidth 0 -height 21 -relheight 0 \
        -anchor nw -bordermode ignore 
    place $top.but43 \
        -in $top -x 1110 -y 30 -width 87 -height 24 -anchor nw \
        -bordermode ignore 
    place $top.but44 \
        -in $top -x 1220 -y 30 -width 87 -height 24 -anchor nw \
        -bordermode ignore 

    vTcl:FireEvent $base <<Ready>>
}

#############################################################################
## Binding tag:  _TopLevel

bind "_TopLevel" <<Create>> {
    if {![info exists _topcount]} {set _topcount 0}; incr _topcount
}
bind "_TopLevel" <<DeleteWindow>> {
    if {[set ::%W::_modal]} {
                vTcl:Toplevel:WidgetProc %W endmodal
            } else {
                destroy %W; if {$_topcount == 0} {exit}
            }
}
bind "_TopLevel" <Destroy> {
    if {[winfo toplevel %W] == "%W"} {incr _topcount -1}
}
#############################################################################
## Binding tag:  _vTclBalloon


if {![info exists vTcl(sourcing)]} {
}

set btop ""
if {$vTcl(borrow)} {
    set btop .bor[expr int([expr rand() * 100])]
    while {[lsearch $btop $vTcl(tops)] != -1} {
        set btop .bor[expr int([expr rand() * 100])]
    }
}
set vTcl(btop) $btop
Window show .
Window show .top42 $btop
if {$vTcl(borrow)} {
    $btop configure -background plum
}

