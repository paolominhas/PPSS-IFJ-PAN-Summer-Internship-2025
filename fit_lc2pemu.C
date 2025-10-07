#include <TH1.h>
#include <TH2.h>
#include <TH3.h>
#include <TChain.h>
#include <TTree.h>
#include <TProfile.h>
#include <TPaveText.h>
#include <TLegend.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <iostream>
#include <fstream>
#include <TPaveLabel.h>
#include <TLine.h>
#include <TGraph.h>
#include <TGraphErrors.h>
#include <TText.h>
#include <TLatex.h>
#include "TMath.h"
#include "TLeaf.h"
#include "TLorentzVector.h"
#include <vector>
#include <string>
#include <sstream>
#ifndef __CINT__
#include "RooGlobalFunc.h"
#endif
#include "RooRealVar.h"
#include "RooDataSet.h"
#include "RooDataHist.h"
#include "RooVoigtian.h"
#include "RooBreitWigner.h"
#include "RooGaussian.h"
#include "RooPolynomial.h"
#include "RooExponential.h"
#include "RooGenericPdf.h"
#include "RooCBShape.h"
#include "RooKeysPdf.h"
#include "RooJohnson.h"
#include "RooExtendPdf.h"
#include "RooAddPdf.h"
#include "RooPlot.h"
#include "RooHist.h"
#include "RooWorkspace.h"
#include "RooFitResult.h"
#include "RooSimultaneous.h"
#include "RooCategory.h"

#include "TVector.h" 
#include "TObjArray.h"
#include "TObjString.h"
#include "TLimit.h"
#include "TLimitDataSource.h"
#include "TConfidenceLevel.h"

using namespace std;
using namespace RooFit;

double fit_cb(TH1F *hist, double mstart, string fitname, string comment, string shape, RooWorkspace* rws ); 
double fit_cb_gaus(TH1F *hist, double mstart, string fitname, string comment, string shape, RooWorkspace* rws );
double fit_cb_exp(TH1F *hist, double mstart, string fitname, string comment, string shape, RooWorkspace* rws );
void draw1hist(TH1 *h1, const char * titax, const char * titay, 
	       const char * namfig, const char * dopt, double norm);

std::pair<TTree*, TTree*> split_ntuple_by_target(const std::string& inputFileName, const std::string& treeName = "myNtuple") {
    
    TFile *inputFile = TFile::Open(inputFileName.c_str(), "READ");
    if (!inputFile || inputFile->IsZombie()) {
        std::cerr << "Error: Could not open input file: " << inputFileName << std::endl;
        return {nullptr, nullptr};
    }

    TTree *inputTree = (TTree*)inputFile->Get(treeName.c_str());
    if (!inputTree) {
        std::cerr << "Error: Could not find TTree '" << treeName << "' in the file." << std::endl;
        inputFile->Close();
        delete inputFile;
        return {nullptr, nullptr};
    }

    // Crucial step: Change the current directory to memory (nullptr).
    // This ensures that the new TTrees created by CopyTree() are owned by the program,
    // not by the inputFile, so they won't be deleted when inputFile is closed.
    gDirectory->cd(nullptr);

    // Create the new TTrees in memory by copying entries.
    TTree *tree_target0 = inputTree->CopyTree("target == 0");
    tree_target0->SetName("ntuple_target0");
    
    TTree *tree_target1 = inputTree->CopyTree("target == 1");
    tree_target1->SetName("ntuple_target1");

    // The original file can now be closed.
    inputFile->Close();
    delete inputFile;

    // Return the pair of newly created TTrees.
    return {tree_target0, tree_target1};
}

void analyze_and_draw_region(TCanvas& canfig, TPad* pad1, RooPlot* mFrame,
                             RooRealVar& mass, RooDataHist& dh,
                             RooExponential& cexpo, RooRealVar& cnbkg,
                             RooAddPdf& g_pcbgs, RooRealVar& cnppipi,
                             const string& base_filename, double center, double half_width) {
    
    // --- 1. Define Signal Region & Calculate ---
    double lower_bound = center - half_width;
    double upper_bound = center + half_width;
    string range_name = "signal_window";
    mass.setRange(range_name.c_str(), lower_bound, upper_bound);

    // Calculate total data events in the defined region
    double total_events_in_region = dh.sum(kTRUE, range_name.c_str());

    // Create integrals of the normalized background shapes over the region
    RooAbsReal* int_expo = cexpo.createIntegral(mass, NormSet(mass), Range(range_name.c_str()));
    RooAbsReal* int_ppipi = g_pcbgs.createIntegral(mass, NormSet(mass), Range(range_name.c_str()));
    
    // Calculate the expected number of background events in the region
    double bkg_in_region = cnbkg.getVal() * int_expo->getVal() + 
                           cnppipi.getVal() * int_ppipi->getVal();
    
    delete int_expo;
    delete int_ppipi;

    // --- 2. Print to Console ---
    cout << "\n--- Signal Region Analysis ---" << endl;
    cout << "Region [" << lower_bound << ", " << upper_bound << "] MeV/c^2" << endl;
    cout << "  - Data Events in Region: " << TString::Format("%.0f", total_events_in_region) << endl;
    cout << "  - Expected Bkg Events in Region: " << TString::Format("%.1f", bkg_in_region) << endl;
    cout << "---------------------------------" << endl;

    // --- 3. Draw on Plot & Save ---
    
    // Create a text box for the results
    TPaveText *pt = new TPaveText(0.55, 0.65, 0.88, 0.85, "NDC");
    pt->SetBorderSize(1);
    pt->SetFillColor(kWhite);
    pt->SetTextAlign(12); // Left-aligned
    pt->SetTextFont(42);
    pt->SetTextSize(0.045);
    pt->AddText(TString::Format("Region: M_{#Lambda_{c}} #pm %.0f MeV", half_width));
    pt->AddText(TString::Format("Data Events: %.0f", total_events_in_region));
    pt->AddText(TString::Format("Bkg Events: %.1f", bkg_in_region));

    // --- Linear scale plot ---
    canfig.cd();
    pad1->cd();
    double y_max = mFrame->GetYaxis()->GetXmax();
    TLine line_low(lower_bound, 0, lower_bound, y_max);
    TLine line_high(upper_bound, 0, upper_bound, y_max);
    line_low.SetLineColor(kRed);
    line_high.SetLineColor(kRed);
    line_low.SetLineStyle(kDashed);
    line_high.SetLineStyle(kDashed);

    // Draw lines and text, using DrawClone to avoid ROOT ownership issues
    line_low.DrawClone("same");
    line_high.DrawClone("same");
    pt->DrawClone("same");
    
    canfig.Update();
    canfig.SaveAs(("figs/" + base_filename + ".pdf").c_str());

    // --- Log scale plot ---
    pad1->SetLogy();
    canfig.Update(); // Update to get new log axis range

    double y_min_log = pad1->GetUymin();
    double y_max_log = pad1->GetUymax();

    // Recreate lines with y-coordinates for the log scale
    TLine line_low_log(lower_bound, y_min_log, lower_bound, y_max_log);
    TLine line_high_log(upper_bound, y_min_log, upper_bound, y_max_log);
    line_low_log.SetLineColor(kRed);
    line_high_log.SetLineColor(kRed);
    line_low_log.SetLineStyle(kDashed);
    line_high_log.SetLineStyle(kDashed);

    line_low_log.DrawClone("same");
    line_high_log.DrawClone("same");
    pt->DrawClone("same"); // TPaveText is in NDC, so it's fine

    canfig.Update();
    canfig.SaveAs(("figs/" + base_filename + "_log.pdf").c_str());

    pad1->SetLogy(0); // Reset pad to linear scale for good practice
    delete pt;
}

void fit_lc2pemu() {
   
  // masses 
  double M_Lc = 2286.46;
  double M_phi_PDG = 1019.455;
  double M_omega_PDG = 782.65;
  double M_rho_PDG = 775.26;
  double M_eta_PDG = 547.86;

  // sideband definitions
  double m_sideband_left_low   = M_Lc - 20*7.; 
  double m_sideband_left_high  = M_Lc - 5*7.; 
   
  double m_sideband_right_high = M_Lc + 20*7.; 
  double m_sideband_right_low = M_Lc + 5*7.; 

  // histogram ranges
  double m_mLeft  = m_sideband_left_low;   
  double m_mRight = m_sideband_right_high;   

  double nBins=100;

  string data_dir="/Users/paolominhas/Desktop/Desktop Files/Academic Work/IFJ-PAN/Project/Final/OutputData/paolo_xgb_data";
  //string data_dir="/Users/paolominhas/Desktop/Desktop Files/Academic Work/IFJ-PAN/Project/FinalDataAnalysis/OutputData/Run5";

  // create figs dir if does not exists yet
  int is_dir = system("test -d figs");
  if ( is_dir != 0 ) {
    cout << "creating figs directory" << endl;;
    system("mkdir figs");
  }

auto result = split_ntuple_by_target(data_dir + "/BDTOutputLabelled.root", "DecayTree");
auto result_ref = split_ntuple_by_target(data_dir + "/BDTOutputRefLabelled.root", "DecayTreeRef");

TTree* tree_signal   = result.first;
TTree* tree_data   = result.second;
TTree* tree_ref_data  = result_ref.first;
TTree* tree_pmumu    = result_ref.second;


  /*
  TChain * tree_signal = new TChain("DecayTree");
  tree_signal->Add( (data_dir+"/BDTOutputLabelled.root").c_str());
   
  TChain * tree_data = new TChain("DecayTree");
  tree_data->Add( (data_dir+"/BDTOutputLabelled.root").c_str());

  TChain * tree_pmumu = new TChain("DecayTree");
  tree_pmumu->Add( (data_dir+"/BDTOutputRefLabelled.root").c_str());
  */

  //TChain * tree_signal = new TChain("DecayTree");
  //tree_signal->Add( (data_dir+"/output_with_bdt.root").c_str());

  // find optimal cut using Punzi FoM
  double bdt_min=0;
  double bdt_max=1.;
  int nbins_bdt=100;
  TH1F * h_eff = new TH1F("h_eff", "efficiency" , nbins_bdt, bdt_min, bdt_max);
  h_eff->Sumw2();
  TH1F * h_bkg = new TH1F("h_bkg", "nevents bkg" , nbins_bdt, bdt_min, bdt_max);
  h_bkg->Sumw2();
  TH1F * h_punzi = new TH1F("h_punzi", "Punzi FoM" , nbins_bdt, bdt_min, bdt_max);
  h_punzi->Sumw2();

  TH1F * h_eff_cumul = new TH1F("h_eff_cumul", "efficiency cumulative" , nbins_bdt, bdt_min, bdt_max);
  h_eff->Sumw2();
  TH1F * h_bkg_cumul = new TH1F("h_bkg_cumul", "nevents bkg cumulative" , nbins_bdt, bdt_min, bdt_max);
  h_bkg->Sumw2();

  tree_signal->Draw("bdt_score>>h_eff", "1", "goff");  
  tree_data->Draw("bdt_score>>h_bkg", "1", "goff");
  double ns_all=h_eff->Integral(1,nbins_bdt);
  double bkg_ratio=50./(m_mRight-m_mLeft);
  for (int ibin=1; ibin<nbins_bdt+1; ibin++) {
    double ns=h_eff->Integral(ibin,nbins_bdt);
    double nb=bkg_ratio*h_bkg->Integral(ibin,nbins_bdt);
    double eff=ns/ns_all;
    h_eff_cumul->SetBinContent(ibin,eff);
    h_bkg_cumul->SetBinContent(ibin,sqrt(nb));
    double fom_punzi=eff/(3./2.+sqrt(nb));
    cout<< "FoM: " << ibin << " " << eff << " " << nb << " " << fom_punzi << endl;
    h_punzi->SetBinContent(ibin,fom_punzi);
  }
  draw1hist(h_punzi, "bdt", "FoM", "fom_punzi", "", 0);
  draw1hist(h_eff_cumul, "bdt_cut", "efficiency", "eff_punzi", "", 0);
  draw1hist(h_bkg_cumul, "bdt_cut", "n bkg", "bkg_punzi", "", 0);
   
  string cut_signal="Lc_MM>0 && bdt_score>0.15"; // DUMMY. fill later with bdt cut

  TH1F * h_SigShape = new TH1F("h_SigShape", "signal pemu mass shape" , nBins, m_mLeft, m_mRight );
  h_SigShape->Sumw2();
  tree_signal->Draw("Lc_MM>>h_SigShape", cut_signal.c_str(), "goff"); 

  TH1F * h_Data = new TH1F("h_Data", "data histogram" , nBins, m_mLeft, m_mRight );
  h_Data->Sumw2();
  tree_data->Draw("Lc_MM>>h_Data", cut_signal.c_str(), "goff"); 
   
  TH1F * h_pmumu = new TH1F("h_pmumu", "data histogram" , nBins, m_mLeft, m_mRight );
  h_pmumu->Sumw2();
  tree_pmumu->Draw("Lc_MM>>h_pmumu", cut_signal.c_str(), "goff"); 
   
  RooWorkspace* wshape_s1 = new RooWorkspace("shape_s1");
  fit_cb(h_SigShape, M_Lc , "SignalShapeCB", "MC: CB Fit to Lc2pemu mass after selection", "DUMMY", wshape_s1);
  RooWorkspace* wshape_s2 = new RooWorkspace("shape_s2");
  fit_cb_gaus(h_SigShape, M_Lc , "SignalShapeCB_Gauss", "MC: CB+Gauss Fit to Lc2pemu mass after selection", "DUMMY", wshape_s2);
  RooWorkspace* w_pmumu = new RooWorkspace("w_pmumu");
  fit_cb_exp(h_pmumu, M_Lc , "Lc2pmumu_data", " Lc2pmumu data", "DUMMY", w_pmumu);
  
  
  RooRealVar mass("mass", "mass", m_mLeft, m_mRight);
  // build data model signal + background
  double mstart=M_Lc;
  RooRealVar mcbmean("mcbmean", "cb mean", mstart, mstart-50., mstart+50.);
  RooRealVar mcbsig("mcbsig", "cb sigma", 8.0 , 5.0, 12.0);  
  RooRealVar mcbalpha("mcbalpha", "cb alpha", 0.3, -0.1, 20. );
  RooRealVar mcbn("mcbn","cb n", 3.0, 0.1, 30.);
  RooCBShape mcb("mcb","crystal ball", mass,mcbmean, mcbsig, mcbalpha, mcbn);

  RooRealVar mgmean("gmean", "gaus mean", mstart-50, mstart-100., mstart+100.);
  RooRealVar mgsig("gsig", "g sigma", 90.0 , 10.0, 150.0); 
  RooGaussian mgs("gs","gauss", mass, mgmean, mgsig);

  RooRealVar mdfrac_val("mdfrac_val","fraction", 0.8, 0., 1. ) ;
  RooAddPdf* mcbgm = new RooAddPdf("mcbgm","cb+gaus",RooArgList(mcb,mgs),mdfrac_val); 
  RooRealVar cnsig("cnsig", "nsig", 50, -20., 900000.);
  RooExtendPdf emcbgm(" emcbgm","extended cb+gauss", *mcbgm, cnsig );
  

  RooWorkspace * rws_signal =  wshape_s2;

  mcbmean.setVal(rws_signal->var("cbmean")->getVal());
  mcbsig.setVal(rws_signal->var("cbsig")->getVal());
  mcbalpha.setVal(rws_signal->var("cbalpha")->getVal());
  mcbn.setVal(rws_signal->var("cbn")->getVal());
  mgmean.setVal(rws_signal->var("gmean")->getVal());
  mgsig.setVal(rws_signal->var("gsig")->getVal());
  mdfrac_val.setVal(rws_signal->var("dfrac_val")->getVal());

  mcbmean.setConstant(kTRUE);
  mcbsig.setConstant(kTRUE);
  mcbalpha.setConstant(kTRUE);
  mcbn.setConstant(kTRUE);
  mgmean.setConstant(kTRUE);
  mgsig.setConstant(kTRUE);
  mdfrac_val.setConstant(kTRUE);

  // ppipi background
  RooRealVar cnppipi("cnppipi", "nbkg ppipi", 30, 0., 900000.);
  RooRealVar g_pcbmean("g_pcbmean", "cb mean ppipi", mstart, mstart-50., mstart+50.);
  RooRealVar g_pcbsig("g_pcbsig", "cb sigma ppipi", 6.0 , 5.0, 12.0);  
  RooRealVar g_pcbalpha("g_pcbalpha", "cb alpha ppipi", 1., -0.1, 20. );
  RooRealVar g_pcbn("g_pcbn","cb n ppipi", 1.9, 0.1, 50.);
  RooCBShape g_pcb("g_pcb","crystal ball", mass, g_pcbmean, g_pcbsig, g_pcbalpha, g_pcbn);

  RooRealVar g_pgmean("g_pgmean", "gaus mean ppipi", mstart, mstart-50., mstart+50.);
  RooRealVar g_pgsig("g_pgsig", "g sigma ppipi", 20.0 , 5.0, 100.0); // max 20
  RooGaussian g_pgs("g_pgs","gauss ppipi", mass, g_pgmean, g_pgsig);

  RooRealVar g_pfrac_val("g_pfrac_val","fraction ppipi", 0.8, 0., 1. ) ;
  RooAddPdf g_pcbgs("g_pcbgs","cb+gaus ppipi",RooArgList(g_pcb,g_pgs),g_pfrac_val); 
  RooExtendPdf eg_pcbgs("eg_pcbgs","extended cb+gauss", g_pcbgs, cnppipi );

  //g_pcbmean.setVal(2.271717e+03);
  g_pcbmean.setVal(2.260e+03);
  g_pcbsig.setVal(7.826174e+00);
  g_pcbalpha.setVal(7.990277e-01);
  g_pcbn.setVal(1.950371e+01);
  g_pgmean.setVal(2.254521e+03);
  g_pgsig.setVal(6.879962e+01);
  g_pfrac_val.setVal(9.508733e-01);
  // fix
  //g_pcbmean.setConstant(kTRUE);
  g_pcbsig.setConstant(kTRUE);
  g_pcbalpha.setConstant(kTRUE);
  g_pcbn.setConstant(kTRUE);
  g_pgmean.setConstant(kTRUE);
  g_pgsig.setConstant(kTRUE);
  g_pfrac_val.setConstant(kTRUE);


  // exponential background
  RooRealVar cnbkg("nbkg", "nbkg", h_Data->GetEntries(), 0., 900000.);
  RooRealVar clambda("clambda", "slope", +0.01, -0.5, 0.5);
  RooExponential cexpo("cexpo", "exponential PDF", mass, clambda);
  RooExtendPdf ceexpo("ceexpo","extended expo",cexpo, cnbkg);
    
  // signal + background model
  RooAddPdf model("model","cbg+cexpo",RooArgList(emcbgm, eg_pcbgs, ceexpo));

  RooDataHist dh("dh","histo dataset s+b", mass, Import(*h_Data));
  RooFitResult* res = model.fitTo(dh,Save());

  cnsig.setVal(0.);        
  cnsig.setConstant(kTRUE);

  // background only model
  RooFitResult* res_b = model.fitTo(dh,Save());
  
  double NLL_sb= res->minNll();
  double NLL_b= res_b->minNll(); 
  
  cout<<"NLL sb,b " <<NLL_sb<<" , "<<NLL_b<<endl;
  double delta=NLL_b-NLL_sb;
  delta=delta*2.;
  double significance=sqrt(delta);
  cout<<"q2rr Significance LogL for " << " = " << significance  << endl;

  //
  // produce figures for S+B model
  //
  RooPlot * mFrame = mass.frame(Bins(nBins));
  mFrame->GetXaxis()->SetTitle("#it{m}(#it{pe#mu}) [MeV/#font[12]{c}^{2}]"); 
  string frame_title="Fit results";
  mFrame->SetTitle(frame_title.c_str()); 
  dh.plotOn(mFrame);
  model.plotOn(mFrame, RooFit::Components(emcbgm), RooFit::LineStyle(kDashed), RooFit::LineColor(2)) ;
  model.plotOn(mFrame, RooFit::Components(ceexpo), RooFit::LineStyle(kDashDotted)) ;
  model.plotOn(mFrame, RooFit::Components(eg_pcbgs), RooFit::LineStyle(kDashed)) ;
  model.plotOn(mFrame);

  RooHist *hresid = mFrame->residHist();
  RooHist *hpull = mFrame->pullHist();

  RooPlot* frame3 = mass.frame();                       
  frame3->addPlotable( hpull, "P" );        
  frame3->GetXaxis()->SetTitle("");                      
  frame3->GetXaxis()->SetLabelSize(0.1);                 
  frame3->GetYaxis()->SetLabelSize(0.1);                 

  frame3->SetTitle("");  

  string fitname="Lc2pemu_mass";
  string cnam="c_"+fitname;
  TCanvas canfig(cnam.c_str(),cnam.c_str(),0,0,600,400);
  
  TPad *pad1 = new TPad("pad1cb", "The pad 80% of the height",0.0, 0.2, 1.0, 1.0, 0);
  TPad *pad2 = new TPad("pad2cb", "The pad 20% of the height",0.0, 0.0, 1.0, 0.2, 0);

  canfig.cd();
  pad1->Draw();
  pad2->Draw();
        
  pad1->cd();
  mFrame->Draw();
  pad1->Draw();
  
  pad2->cd();
  frame3->Draw();
  pad2->Draw();
  canfig.Update();

  /*
  canfig.SaveAs( ("figs/"+fitname+".pdf").c_str() );
  //canfig.SaveAs( ("figs/"+fitname+".jpg").c_str() );
  pad1->SetLogy();
  canfig.Update();
  canfig.SaveAs( ("figs/"+fitname+"_log.pdf").c_str() );
  */

  analyze_and_draw_region(canfig, pad1, mFrame, mass, dh, cexpo, cnbkg, g_pcbgs, cnppipi, fitname, 2286.46, 15.0);
  

  
}

//================================================================================
double fit_cb(TH1F *hist, double mstart, string fitname, string comment, string shape, RooWorkspace* rws) {

  double  m1 = hist->GetXaxis()->GetXmin();
  double  m2 = hist->GetXaxis()->GetXmax();
  RooRealVar mass("mass", "mass", m1, m2);

  RooRealVar nsig("nsig", "nsig", 3000, 0., 900000.);

  RooRealVar cbmean("cbmean", "cb mean", mstart, mstart-50., mstart+50.);
  RooRealVar cbsig("cbsig", "cb sigma", 15.0 , 1.0, 30.0);
  RooRealVar cbalpha("cbalpha", "cb alpha", 1.5, 1., 5. );
  RooRealVar cbn("cbn","cb n", 1., 0.1, 10.);
  RooCBShape cb("cb","crystal ball", mass, cbmean, cbsig, cbalpha, cbn);

  RooExtendPdf model(fitname.c_str(),"extended cb",cb, nsig);

  RooPlot * mFrame = mass.frame();
  mFrame->SetTitle("");
  RooDataHist  dh(fitname.c_str(),fitname.c_str(),mass, Import(*hist));
  model.fitTo(dh);

  //  RooPlot * mFrame = mass.frame();
  dh.plotOn(mFrame);
  model.plotOn(mFrame);

  RooHist *hresid = mFrame->residHist();
  RooHist *hpull = mFrame->pullHist();

  RooPlot* frame3 = mass.frame();                       
  frame3->addPlotable( hpull, "P" );        
  frame3->GetXaxis()->SetTitle("");                      
  frame3->GetXaxis()->SetLabelSize(0.1);                 
  frame3->GetYaxis()->SetLabelSize(0.1);                 

  frame3->SetTitle("");  

  string cnam="c_"+fitname;
  TCanvas canfig(cnam.c_str(),cnam.c_str(),0,0,600,400);
  TPad *pad1 = new TPad("pad1cb", "The pad 80% of the height",0.0, 0.2, 1.0, 1.0,0);
  TPad *pad2 = new TPad("pad2cb", "The pad 20% of the height",0.0,0.0,1.0,0.2,0);

  canfig.cd();
  pad1->Draw();
  pad2->Draw();
        
  pad1->cd();
  mFrame->Draw();
  pad1->Draw();
  
  pad2->cd();
  frame3->Draw();
  pad2->Draw();
  canfig.Update();


  string fnam=fitname+".pdf";
  //  mFrame->Draw();
  canfig.SaveAs( ("figs/"+fitname+".pdf").c_str() );
  pad1->SetLogy();
  canfig.Update();
  canfig.SaveAs( ("figs/"+fitname+"_log.pdf").c_str() );

  cout << endl << endl << "cb " << " / " << fitname << " / " << comment << endl;
  cout << "================================================================================" << endl;
  cout << " n signal = " << nsig.getVal()    <<  " +-  " << nsig.getError() << endl;
  cout << " cbmean     = " << cbmean.getVal()  <<  " +-  " << cbmean.getError() << endl; 
  cout << " sigma    = " << cbsig.getVal()   <<  " +-  " << cbsig.getError() << endl;
  cout << " cbn      = " << cbn.getVal()     <<  " +-  " << cbn.getError() << endl;
  cout << " cbalpha  = " << cbalpha.getVal() <<  " +-  " << cbalpha.getError() << endl << endl;

  if (rws != 0) rws->import(model);

  return nsig.getVal();
}


double fit_cb_gaus(TH1F *hist, double mstart, string fitname, string comment, string shape, RooWorkspace* rws) {

  double  m1 = hist->GetXaxis()->GetXmin();
  double  m2 = hist->GetXaxis()->GetXmax();
  int nbins = hist->GetNbinsX();
  RooRealVar mass("mass", "mass", m1, m2);

  RooRealVar nsig("nsig", "nsig", 3000, 0., 900000.);

  //  RooRealVar cbmean("cbmean", "cb mean", 2.27258e+03, 2220., 2320.);
  RooRealVar cbmean("cbmean", "cb mean", mstart, mstart-50., mstart+50.);
  RooRealVar cbsig("cbsig", "cb sigma", 8.0 , 5.0, 12.0);  
  RooRealVar cbalpha("cbalpha", "cb alpha", 0.3, -0.1, 20. );
  RooRealVar cbn("cbn","cb n", 3.0, 0.1, 30.);
  RooCBShape cb("cb","crystal ball", mass,cbmean, cbsig, cbalpha, cbn);

  RooRealVar gmean("gmean", "gaus mean", mstart-50, mstart-100., mstart+100.);
  RooRealVar gsig("gsig", "g sigma", 90.0 , 10.0, 150.0); 
  RooGaussian gs("gs","gauss", mass, gmean, gsig);

  RooRealVar dfrac_val("dfrac_val","fraction", 0.8, 0., 1. ) ;
  RooAddPdf* cbgm = new RooAddPdf("cbgm","cb+gaus",RooArgList(cb,gs),dfrac_val); 


  RooExtendPdf model(fitname.c_str(),"extended cb",*cbgm, nsig);

  RooPlot * mFrame = mass.frame(Bins(nbins));
  mFrame->SetTitle("");
  RooDataHist  dh(fitname.c_str(),fitname.c_str(),mass, Import(*hist));
  model.fitTo(dh);

  //  RooPlot * mFrame = mass.frame();
  dh.plotOn(mFrame);
  model.plotOn(mFrame);

  RooHist *hresid = mFrame->residHist();
  RooHist *hpull = mFrame->pullHist();

  RooPlot* frame3 = mass.frame();                       
  frame3->addPlotable( hpull, "P" );        
  frame3->GetXaxis()->SetTitle("");                      
  frame3->GetXaxis()->SetLabelSize(0.1);                 
  frame3->GetYaxis()->SetLabelSize(0.1);                 

  frame3->SetTitle("");  

  string cnam="c_"+fitname;
  TCanvas canfig(cnam.c_str(),cnam.c_str(),0,0,600,400);
  TPad *pad1 = new TPad("pad1cb", "The pad 80% of the height",0.0, 0.2, 1.0, 1.0,0);
  TPad *pad2 = new TPad("pad2cb", "The pad 20% of the height",0.0,0.0,1.0,0.2,0);

  canfig.cd();
  pad1->Draw();
  pad2->Draw();
        
  pad1->cd();
  mFrame->Draw();
  pad1->Draw();
  
  pad2->cd();
  frame3->Draw();
  pad2->Draw();
  canfig.Update();


  string fnam=fitname+".pdf";
  //  mFrame->Draw();
  canfig.SaveAs( ("figs/"+fitname+".pdf").c_str() );
  pad1->SetLogy();
  canfig.Update();
  canfig.SaveAs( ("figs/"+fitname+"_log.pdf").c_str() );

  
  cout << endl << endl << "cb+gaus " << " / " << fitname << " / " << comment << endl;
  cout << "================================================================================" << endl;
  cout << " n signal  = " << nsig.getVal()     <<  " +-  " << nsig.getError() << endl;
  cout << " cbmean    = " << cbmean.getVal()   <<  " +-  " << cbmean.getError() << endl; 
  cout << " cbsig    = " << cbsig.getVal()   <<  " +-  " << cbsig.getError() << endl;
  cout << " cbn      = " << cbn.getVal()     <<  " +-  " << cbn.getError() << endl;
  cout << " cbalpha  = " << cbalpha.getVal() <<  " +-  " << cbalpha.getError() << endl << endl;
  cout << " gmean    = " << gmean.getVal()   <<  " +-  " << gmean.getError() << endl; 
  cout << " gsig    = " << gsig.getVal()   <<  " +-  " << gsig.getError() << endl;
  cout << " dfrac_val = " << dfrac_val.getVal() << " +- "  << dfrac_val.getError() << endl << endl;
  
  if (rws != 0) rws->import(model);

  return nsig.getVal();
}

//================================================================================
double fit_cb_exp(TH1F *hist, double mstart, string fitname, string comment, string shape, RooWorkspace* rws) {

  double  m1 = hist->GetXaxis()->GetXmin();
  double  m2 = hist->GetXaxis()->GetXmax();
  int nbins = hist->GetNbinsX();
  RooRealVar masscb("masscb", "mass", m1, m2);

  masscb.setRange("winsig",mstart-20., mstart+20.);  // +-3 sigma around Lc mass
  masscb.setRange("winfull",m1, m2);  // full range

  RooRealVar nbkg("nbkg", "nbkg", 100, 0., 900000.);
  RooRealVar nsig("nsig", "nsig", 100, 0., 900000.);

  RooRealVar cbmean("cbmean", "cb mean", 2286.46, 2220., 2320.);
  RooRealVar cbsig("cbsig", "cb sigma", 8.0 , 4.0, 15.0);
  RooRealVar cbalpha("cbalpha", "cb alpha", 1.5, 1., 5. );
  RooRealVar cbn("cbn","cb n", 1.9, 0.1, 10.);
  RooCBShape cb("cb","crystal ball", masscb, cbmean, cbsig, cbalpha, cbn);

  RooRealVar lambda("lambda", "slope", +0.01, -0.1, 0.1);
  RooExponential expo("expo", "exponential PDF", masscb, lambda);

  RooExtendPdf ecb("ecb","extended cb",cb, nsig);
  RooExtendPdf eexpo("eexpo","extended expo",expo, nbkg);

  RooAddPdf  model(fitname.c_str(),"cb+expo",RooArgList(ecb,eexpo));

  // use predefined shape if required
  if ( shape == "PHIPRESEL" ) {
    cbsig.setVal(7.779913e+00);
    cbalpha.setVal(2.147048e+00);
    cbn.setVal(2.434411e+00);
    cbsig.setConstant(kTRUE);
    cbalpha.setConstant(kTRUE);
    cbn.setConstant(kTRUE);
  }

  if ( shape == "PHISEL" ) {
    //WWWW if uncommented default is floating the mean
    cbmean.setVal(2.288075e+03);
    cbmean.setConstant(kTRUE);
    //WWWW
    cbsig.setVal(7.379936e+00);
    cbalpha.setVal(2.020361e+00);
    cbn.setVal(2.774594e+0);
    cbsig.setConstant(kTRUE);
    cbalpha.setConstant(kTRUE);
    cbn.setConstant(kTRUE);
  }

  if ( shape == "PHIFREE" ) {
    cbmean.setVal(2.288075e+03);
    cbsig.setVal(7.379936e+00);
    cbalpha.setVal(2.020361e+00);
    cbn.setVal(2.774594e+0);
    //cbsig.setConstant(kTRUE);
    //cbalpha.setConstant(kTRUE);
    //cbn.setConstant(kTRUE);
  }


  RooPlot * mFrame = masscb.frame(Bins(nbins));
  mFrame->SetTitle("");
  mFrame->GetXaxis()->SetTitle("#it{m}(#it{p#mu^{#plus}#mu^{#minus}}) [MeV/#font[12]{c}^{2}]");
  if (false) {
   TLatex *txt9 = new TLatex(0.13,0.7,"|m(#phi)-m(#mu#mu)|<40 MeV/#font[12]{c}^{2}");
   txt9->SetNDC();
   txt9->SetTextSize(0.05);
   mFrame->addObject(txt9);
  }

  RooDataHist  dh(fitname.c_str(),fitname.c_str(),masscb, Import(*hist));
  model.fitTo(dh);
  RooAbsReal* bkgint = eexpo.createIntegral(masscb,NormSet(masscb),Range("winsig"));

  // *******************************************************************************************************************
  // WARNING ORDER OF model.plotOn is IMPORTANT. 
  // PULL distribution is calculated wrt last plotOn !!!!!!!!!!!!!!!!!!!!!!!!!
  // *******************************************************************************************************************

  dh.plotOn(mFrame, DataError(RooAbsData::SumW2));
  model.plotOn(mFrame, RooFit::Components(eexpo), RooFit::LineStyle(kDashed)) ;
  model.plotOn(mFrame);

  RooHist *hresid = mFrame->residHist();
  RooHist *hpull = mFrame->pullHist();

  string cnam="c_"+fitname;
  TCanvas canfig(cnam.c_str(),cnam.c_str(),0,0,600,400);
  TPad *pad1 = new TPad("pad1u", "The pad 80% of the height",0.0, 0.2, 1.0, 1.0,0);
  TPad *pad2 = new TPad("pad2u", "The pad 20% of the height",0.0,0.0,1.0,0.2,0);
 
  RooPlot* frame3 = masscb.frame();                       
  frame3->addPlotable(hpull, "P" );        
  frame3->GetXaxis()->SetTitle("");                      
  frame3->GetXaxis()->SetLabelSize(0.1);                 
  frame3->GetYaxis()->SetLabelSize(0.1);                 
  frame3->SetTitle("");  

  canfig.cd();
  pad1->Draw();
  pad2->Draw();
        
  pad1->cd();
  mFrame->Draw();
  pad1->Draw();
  
  pad2->cd();
  frame3->Draw();
  pad2->Draw();

  canfig.Update();

  canfig.SaveAs( ("figs/"+fitname+".pdf").c_str() );
  pad1->SetLogy();
  canfig.Update();
  canfig.SaveAs( ("figs/"+fitname+"_log.pdf").c_str() );

  cout << endl << endl << "cb+expo " << " / " << fitname << " / " << comment << endl;
  cout << "================================================================================" << endl;
  cout << " n signal = " << nsig.getVal()    <<  " +-  " << nsig.getError() << endl;
  cout << " mean     = " << cbmean.getVal()  <<  " +-  " << cbmean.getError() << endl; 
  cout << " sigma    = " << cbsig.getVal()   <<  " +-  " << cbsig.getError() << endl;
  cout << " cbn      = " << cbn.getVal()     <<  " +-  " << cbn.getError() << endl;
  cout << " cbalpha  = " << cbalpha.getVal() <<  " +-  " << cbalpha.getError() << endl << endl;

  if (rws != 0) rws->import(model);

  return nsig.getVal();
}

void draw1hist(TH1 *h1, const char * titax, const char * titay, 
                      const char * namfig, const char * dopt, double norm) {
   h1->GetXaxis()->SetTitle(titax);
   h1->GetYaxis()->SetTitle(titay);
   h1->SetStats(0);

   if ( norm>10.E-9 ) {
     h1->Scale(norm/h1->Integral());
   }   
   double dx=600;
   double dy=400;

   if ( string(dopt) == "xwide" ) {
     dx=1200;;
   }
   TCanvas *c01 = new TCanvas(namfig,"c01",0,0,dx,dy);
   if ( string(dopt) == "logy" ) {
     c01->SetLogy();
   }
   if ( string(dopt) == "colz" ) {
      h1->Draw("COLZ");
   } else {
     h1->Draw("");
   }

   string fnam_pdf =  string(namfig)+".pdf";
   string fnam_jpg =  string(namfig)+".jpg";
   c01->SaveAs(("figs/"+fnam_pdf).c_str());   
   c01->SaveAs(("figs/"+fnam_jpg).c_str());   

}
